import tkinter as tk
import threading
import time
import collections
import tkinter.font as tkFont
import re
from tkinter import colorchooser, messagebox
import json

class ClosedCaptionOverlay:
    def __init__(self):
        self.root = None
        self.label = None
        self.text_buffer = ""
        self.is_visible = False
        self.overlay_thread = None
        self.text_lock = threading.Lock()
        self.current_font = ("Arial", 14, "bold")
        self.overlay_width = 500
        self.max_chars_per_line_estimate = 60
        self.display_lines_queue = collections.deque(maxlen=2)
        self._last_displayed_content = ""
        
        # Timer for hiding captions after silence
        self.last_update_time = time.time()
        self.last_sentence_time = time.time()
        self.normal_hide_timeout = 6.0
        self.translation_hide_timeout = 10.0
        self.hide_timeout_seconds = self.normal_hide_timeout
        self.silence_check_interval_ms = 100
        
        # Typewriter effect variables
        self.typewriter_words = []
        self.typewriter_target_words = []
        self.typewriter_index = 0
        self.typewriter_speed = 100
        self.typewriter_job = None
        self.ms_per_word = 200
        self.sentence_gap_threshold = 2.0
        self.showing_translation = False
        self.pending_sentences = []
        
        # Customization settings
        self.text_color = 'white'
        self.bg_color = 'black'
        self.has_background = True
        self.font_family = 'Arial'
        self.font_weight = 'bold'
        self.window_alpha = 0.7
        
        # TTS Configuration
        self.tts_host = 'localhost'
        self.tts_port = '7778'
        self.tts_params_template = {
            'model': 'chatterbox',
            'voice': '',  # Will be filled dynamically
            'speed': 1.0
        }
        self.tts_sentence_concat = 1  # Number of sentences to concatenate before sending to TTS
        
        # Character correction map
        self.character_corrections = {
            'ÃƒÂ±': 'ñ',
            'Ð¶': 'ж', 'Ñ\x86': 'ц', 'Ñ\x8d': 'э', 'Ñ\x85': 'х', 'Ñ„': 'ф',
            'Ð°': 'а', 'Ð±': 'б', 'Ð²': 'в', 'Ð³': 'г', 'Ð´': 'д', 'Ðµ': 'е',
            'Ð·': 'з', 'Ð¸': 'и', 'Ð¹': 'й', 'Ðº': 'к', 'Ð»': 'л', 'Ð¼': 'м',
            'Ð½': 'н', 'Ð¾': 'о', 'Ð¿': 'п', 'Ñ\x80': 'р', 'Ñ\x81': 'с',
            'Ñ\x82': 'т', 'Ñ\x83': 'у', 'Ñ\x84': 'ф', 'Ñ\x85': 'х', 'Ñ\x86': 'ц',
            'Ñ\x87': 'ч', 'Ñ\x88': 'ш', 'Ñ\x89': 'щ', 'Ñ\x8a': 'ъ', 'Ñ\x8b': 'ы',
            'Ñ\x8c': 'ь', 'Ñ\x8d': 'э', 'Ñ\x8e': 'ю', 'Ñ\x8f': 'я',
            'Ð•': 'Е', 'Ð¡': 'С', 'Ð£': 'У', 'Ð¯': 'Я'
        }

    def _show_context_menu(self, event):
        """Show right-click context menu for customization"""
        menu = tk.Menu(self.root, tearoff=0)
        
        # Text Color submenu
        color_menu = tk.Menu(menu, tearoff=0)
        color_menu.add_command(label="White", command=lambda: self._change_text_color('white'))
        color_menu.add_command(label="Yellow", command=lambda: self._change_text_color('yellow'))
        color_menu.add_command(label="Cyan", command=lambda: self._change_text_color('cyan'))
        color_menu.add_command(label="Green", command=lambda: self._change_text_color('green'))
        color_menu.add_command(label="Red", command=lambda: self._change_text_color('red'))
        color_menu.add_separator()
        color_menu.add_command(label="Custom Color...", command=self._choose_text_color)
        menu.add_cascade(label="Text Color", menu=color_menu)
        
        # Background submenu
        bg_menu = tk.Menu(menu, tearoff=0)
        bg_menu.add_checkbutton(label="Show Background", 
                                command=self._toggle_background,
                                variable=tk.BooleanVar(value=self.has_background))
        bg_menu.add_separator()
        bg_menu.add_command(label="Black Background", command=lambda: self._change_bg_color('black'))
        bg_menu.add_command(label="Dark Gray Background", command=lambda: self._change_bg_color('#333333'))
        bg_menu.add_command(label="Custom Background...", command=self._choose_bg_color)
        menu.add_cascade(label="Background", menu=bg_menu)
        
        # Font submenu
        font_menu = tk.Menu(menu, tearoff=0)
        font_menu.add_command(label="Arial", command=lambda: self._change_font('Arial'))
        font_menu.add_command(label="Helvetica", command=lambda: self._change_font('Helvetica'))
        font_menu.add_command(label="Times New Roman", command=lambda: self._change_font('Times New Roman'))
        font_menu.add_command(label="Courier New", command=lambda: self._change_font('Courier New'))
        font_menu.add_command(label="Verdana", command=lambda: self._change_font('Verdana'))
        font_menu.add_separator()
        font_menu.add_checkbutton(label="Bold", 
                                  command=self._toggle_bold,
                                  variable=tk.BooleanVar(value=self.font_weight=='bold'))
        menu.add_cascade(label="Font", menu=font_menu)
        
        # Font Size submenu
        size_menu = tk.Menu(menu, tearoff=0)
        size_menu.add_command(label="10px", command=lambda: self._change_font_size(10))
        size_menu.add_command(label="12px", command=lambda: self._change_font_size(12))
        size_menu.add_command(label="14px", command=lambda: self._change_font_size(14))
        size_menu.add_command(label="16px", command=lambda: self._change_font_size(16))
        size_menu.add_command(label="18px", command=lambda: self._change_font_size(18))
        size_menu.add_command(label="20px", command=lambda: self._change_font_size(20))
        size_menu.add_command(label="24px", command=lambda: self._change_font_size(24))
        size_menu.add_command(label="28px", command=lambda: self._change_font_size(28))
        size_menu.add_command(label="32px", command=lambda: self._change_font_size(32))
        menu.add_cascade(label="Font Size", menu=size_menu)
        
        # Duration submenu
        duration_menu = tk.Menu(menu, tearoff=0)
        
        # Normal mode duration
        normal_duration_menu = tk.Menu(duration_menu, tearoff=0)
        normal_duration_menu.add_command(label="2 seconds", command=lambda: self._set_normal_duration(2.0))
        normal_duration_menu.add_command(label="3 seconds", command=lambda: self._set_normal_duration(3.0))
        normal_duration_menu.add_command(label="4 seconds", command=lambda: self._set_normal_duration(4.0))
        normal_duration_menu.add_command(label="5 seconds", command=lambda: self._set_normal_duration(5.0))
        normal_duration_menu.add_command(label="10 second", command=lambda: self._set_normal_duration(10.0))
        normal_duration_menu.add_command(label="15 seconds", command=lambda: self._set_normal_duration(15.0))
        duration_menu.add_cascade(label="Normal Mode", menu=normal_duration_menu)
        
        # Translation mode duration
        translation_duration_menu = tk.Menu(duration_menu, tearoff=0)
        translation_duration_menu.add_command(label="3 seconds", command=lambda: self._set_translation_duration(3.0))
        translation_duration_menu.add_command(label="5 seconds", command=lambda: self._set_translation_duration(5.0))
        translation_duration_menu.add_command(label="8 seconds", command=lambda: self._set_translation_duration(8.0))
        translation_duration_menu.add_command(label="10 seconds", command=lambda: self._set_translation_duration(10.0))
        translation_duration_menu.add_command(label="15 seconds", command=lambda: self._set_translation_duration(15.0))
        translation_duration_menu.add_command(label="20 seconds", command=lambda: self._set_translation_duration(20.0))
        duration_menu.add_cascade(label="Translation Mode", menu=translation_duration_menu)
        
        # Show current settings
        duration_menu.add_separator()
        duration_menu.add_command(label=f"Normal: {self.normal_hide_timeout}s | Translation: {self.translation_hide_timeout}s", state='disabled')
        
        menu.add_cascade(label="Duration", menu=duration_menu)
        
        # TTS Configuration submenu
        tts_menu = tk.Menu(menu, tearoff=0)
        tts_menu.add_command(label="Edit TTS Parameters...", command=self._edit_tts_params)
        tts_menu.add_command(label="Reset to Default", command=self._reset_tts_params)
        tts_menu.add_separator()
        current_host_port = f"{self.tts_host}:{self.tts_port}"
        tts_menu.add_command(label=f"Host: {current_host_port}", state='disabled')
        menu.add_cascade(label="TTS Config", menu=tts_menu)
        
        menu.tk_popup(event.x_root, event.y_root)

    def _change_text_color(self, color):
        """Change the text color"""
        self.text_color = color
        if self.label:
            self.label.config(fg=color)

    def _choose_text_color(self):
        """Open color picker for text color"""
        color = colorchooser.askcolor(title="Choose Text Color", initialcolor=self.text_color)
        if color[1]:
            self._change_text_color(color[1])

    def _change_bg_color(self, color):
        """Change the background color"""
        self.bg_color = color
        if self.has_background:
            if self.root:
                self.root.config(bg=color)
            if self.frame:
                self.frame.config(bg=color)
            if self.label:
                self.label.config(bg=color)

    def _choose_bg_color(self):
        """Open color picker for background color"""
        color = colorchooser.askcolor(title="Choose Background Color", initialcolor=self.bg_color)
        if color[1]:
            self._change_bg_color(color[1])

    def _toggle_background(self):
        """Toggle background on/off"""
        self.has_background = not self.has_background
        
        if self.has_background:
            # Restore background
            if self.root:
                self.root.config(bg=self.bg_color)
                self.root.attributes('-transparentcolor', '')
            if self.frame:
                self.frame.config(bg=self.bg_color)
            if self.label:
                self.label.config(bg=self.bg_color)
        else:
            # Remove background (make transparent)
            transparent_color = '#000001'  # Nearly black but unique for transparency
            if self.root:
                self.root.config(bg=transparent_color)
                self.root.attributes('-transparentcolor', transparent_color)
            if self.frame:
                self.frame.config(bg=transparent_color)
            if self.label:
                self.label.config(bg=transparent_color)

    def _change_font(self, font_family):
        """Change the font family"""
        self.font_family = font_family
        size = self.current_font[1]
        self.current_font = (font_family, size, self.font_weight)
        if self.label:
            self.label.config(font=self.current_font)
        self._update_max_chars_per_line_estimate()

    def _toggle_bold(self):
        """Toggle bold font weight"""
        self.font_weight = 'normal' if self.font_weight == 'bold' else 'bold'
        size = self.current_font[1]
        self.current_font = (self.font_family, size, self.font_weight)
        if self.label:
            self.label.config(font=self.current_font)

    def _change_font_size(self, size):
        """Change the font size"""
        self.set_font_size(size)

    def _set_normal_duration(self, duration):
        """Set the normal mode caption duration"""
        self.normal_hide_timeout = duration
        print(f"Normal caption duration set to {duration} seconds")

    def _set_translation_duration(self, duration):
        """Set the translation mode caption duration"""
        self.translation_hide_timeout = duration
        print(f"Translation caption duration set to {duration} seconds")

    def _edit_tts_params(self):
        """Open dialog to edit TTS parameters with individual text boxes"""
        # Create a dialog for parameter editing
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit TTS Parameters")
        dialog.geometry("550x650")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create a scrollable frame
        canvas = tk.Canvas(dialog)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Main parameters
        tk.Label(scrollable_frame, text="Main Parameters", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        tk.Label(scrollable_frame, text="Model:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
        model_entry = tk.Entry(scrollable_frame, width=30)
        model_entry.insert(0, self.tts_params_template.get('model', 'chatterbox'))
        model_entry.grid(row=1, column=1, padx=10, pady=5)
        
        tk.Label(scrollable_frame, text="Voice:").grid(row=2, column=0, padx=10, pady=5, sticky='e')
        voice_entry = tk.Entry(scrollable_frame, width=30)
        voice_entry.insert(0, self.tts_params_template.get('voice', ''))
        voice_entry.grid(row=2, column=1, padx=10, pady=5)
        
        tk.Label(scrollable_frame, text="Speed:").grid(row=3, column=0, padx=10, pady=5, sticky='e')
        speed_entry = tk.Entry(scrollable_frame, width=30)
        speed_entry.insert(0, str(self.tts_params_template.get('speed', 1.0)))
        speed_entry.grid(row=3, column=1, padx=10, pady=5)
        
        # Server configuration
        tk.Label(scrollable_frame, text="Server Configuration", font=("Arial", 12, "bold")).grid(row=4, column=0, columnspan=2, pady=(20, 10))
        
        tk.Label(scrollable_frame, text="Host:").grid(row=5, column=0, padx=10, pady=5, sticky='e')
        host_entry = tk.Entry(scrollable_frame, width=30)
        host_entry.insert(0, self.tts_host)
        host_entry.grid(row=5, column=1, padx=10, pady=5)
        
        tk.Label(scrollable_frame, text="Port:").grid(row=6, column=0, padx=10, pady=5, sticky='e')
        port_entry = tk.Entry(scrollable_frame, width=30)
        port_entry.insert(0, self.tts_port)
        port_entry.grid(row=6, column=1, padx=10, pady=5)
        
        # Extra body parameters
        tk.Label(scrollable_frame, text="Extra Body Parameters", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=2, pady=(20, 10))
        
        extra_body = self.tts_params_template.get('extra_body', {})
        params = extra_body.get('params', {})
        
        tk.Label(scrollable_frame, text="Language ID:").grid(row=8, column=0, padx=10, pady=5, sticky='e')
        lang_entry = tk.Entry(scrollable_frame, width=30)
        lang_entry.insert(0, params.get('language_id', ''))
        lang_entry.grid(row=8, column=1, padx=10, pady=5)
        
        tk.Label(scrollable_frame, text="Model Name:").grid(row=9, column=0, padx=10, pady=5, sticky='e')
        model_name_entry = tk.Entry(scrollable_frame, width=30)
        model_name_entry.insert(0, params.get('model_name', ''))
        model_name_entry.grid(row=9, column=1, padx=10, pady=5)
        
        tk.Label(scrollable_frame, text="Clone:").grid(row=10, column=0, padx=10, pady=5, sticky='e')
        gpu_entry = tk.Entry(scrollable_frame, width=30)
        gpu_entry.insert(0, str(params.get('clone', '')))
        gpu_entry.grid(row=10, column=1, padx=10, pady=5)
        
        tk.Label(scrollable_frame, text="last_predefined_voice:").grid(row=11, column=0, padx=10, pady=5, sticky='e')
        multilingual_entry = tk.Entry(scrollable_frame, width=30)
        multilingual_entry.insert(0, str(params.get('last_predefined_voice', '')))
        multilingual_entry.grid(row=11, column=1, padx=10, pady=5)
        
        # TTS Processing settings
        tk.Label(scrollable_frame, text="TTS Processing", font=("Arial", 12, "bold")).grid(row=12, column=0, columnspan=2, pady=(20, 10))
        
        tk.Label(scrollable_frame, text="Sentence Concatenation:").grid(row=13, column=0, padx=10, pady=5, sticky='e')
        concat_entry = tk.Entry(scrollable_frame, width=30)
        concat_entry.insert(0, str(self.tts_sentence_concat))
        concat_entry.grid(row=13, column=1, padx=10, pady=5)
        tk.Label(scrollable_frame, text="(Number of sentences to combine)", font=("Arial", 8)).grid(row=14, column=1, padx=10, pady=0, sticky='w')
        
        def save_params():
            try:
                # Update host and port
                new_host = host_entry.get().strip()
                new_port = port_entry.get().strip()
                
                if new_host:
                    self.tts_host = new_host
                    print(f"TTS host updated to: {self.tts_host}")
                if new_port:
                    self.tts_port = new_port
                    print(f"TTS port updated to: {self.tts_port}")
                
                # Build the new parameters dictionary
                new_params = {
                    'model': model_entry.get().strip() or 'chatterbox',
                    'voice': voice_entry.get().strip(),
                    'speed': float(speed_entry.get().strip() or '1.0'),
                }
                
                # Build extra_body parameters - only include non-empty fields
                extra_params = {}
                
                # Language ID - only add if not empty
                lang_value = lang_entry.get().strip()
                if lang_value:
                    extra_params['language_id'] = lang_value
                
                # Model Name - only add if not empty
                model_name_value = model_name_entry.get().strip()
                if model_name_value:
                    extra_params['model_name'] = model_name_value
                
                # Clone - only add if not empty
                clone_value = gpu_entry.get().strip()
                if clone_value:
                    extra_params['clone'] = clone_value.lower() == 'true'
                
                # last_reference_file - only add if not empty
                last_ref_value = multilingual_entry.get().strip()
                if last_ref_value:
                    extra_params['last_predefined_voice'] = last_ref_value
                
                # Update sentence concatenation setting
                try:
                    new_concat = int(concat_entry.get().strip())
                    if new_concat >= 1 and new_concat <= 10:  # Reasonable limits
                        self.tts_sentence_concat = new_concat
                        print(f"Sentence concatenation updated to: {self.tts_sentence_concat}")
                    else:
                        raise ValueError("Sentence concatenation must be between 1 and 10")
                except ValueError as e:
                    messagebox.showerror("Error", f"Invalid sentence concatenation value: {str(e)}")
                    return
                
                # Only add extra_body if there are parameters
                if extra_params:
                    new_params['extra_body'] = {
                        'params': extra_params
                    }
                
                # Validate basic structure
                if new_params['model'] and isinstance(new_params['speed'], (int, float)):
                    self.tts_params_template = new_params
                    print("TTS parameters updated successfully")
                    print(f"New template: {new_params}")
                    print(f"Host: {self.tts_host}, Port: {self.tts_port}")
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "Invalid parameters: model must be non-empty and speed must be numeric")
                    
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid value: {str(e)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving parameters: {str(e)}")
        
        def cancel():
            dialog.destroy()
        
        # Button frame
        button_frame = tk.Frame(scrollable_frame)
        button_frame.grid(row=15, column=0, columnspan=2, pady=20)
        
        tk.Button(button_frame, text="Save", command=save_params).pack(side='left', padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel).pack(side='left', padx=5)
        
        # Pack the scrollable frame
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _reset_tts_params(self):
        """Reset TTS parameters to default values"""
        self.tts_params_template = {
            'model': 'chatterbox',
            'voice': '',
            'speed': 1.0
        }
        self.tts_host = 'localhost'
        self.tts_port = '7778'
        self.tts_sentence_concat = 1
        print("TTS parameters reset to default")
        print(f"Host: {self.tts_host}, Port: {self.tts_port}, Sentence Concat: {self.tts_sentence_concat}")

    def get_tts_config(self):
        """Get current TTS configuration"""
        return {
            'host': self.tts_host,
            'port': self.tts_port,
            'params': self.tts_params_template
        }

    def _on_mouse_wheel(self, event):
        """Adjust window opacity based on mouse wheel movement"""
        current_alpha = float(self.root.attributes('-alpha'))
        if event.delta > 0:
            new_alpha = min(1.0, current_alpha + 0.05)
        else:
            new_alpha = max(0.05, current_alpha - 0.05)
        self.window_alpha = new_alpha
        self.root.attributes('-alpha', new_alpha)

    def _run_tkinter(self):
        """Run the tkinter main loop in a separate thread"""
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Closed Captions")
        self.root.configure(bg=self.bg_color)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', self.window_alpha)
        self.root.overrideredirect(True)
        
        # Bind mouse events
        self.root.bind('<MouseWheel>', self._on_mouse_wheel)
        self.root.bind('<Button-3>', self._show_context_menu)  # Right-click
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.overlay_width = min(600, int(screen_width * 0.3))
        overlay_height = int(screen_height * 0.06)
        self.set_font_size(max(16, int(screen_height * 0.02)))
        
        self.frame = tk.Frame(self.root, bg=self.bg_color)
        self.frame.pack(fill='both', expand=True)
        self.frame.bind('<Button-3>', self._show_context_menu)
        
        self.label = tk.Label(
            self.frame,
            text="",
            font=self.current_font,
            fg=self.text_color,
            bg=self.bg_color,
            wraplength=self.overlay_width,
            anchor='w',
            justify='left',
            padx=10,
            pady=5,
            bd=0,
            highlightthickness=0
        )
        self.label.pack(side='left', fill='both', expand=True)
        self.label.bind('<Button-3>', self._show_context_menu)
        
        self.root.bind('<Button-1>', self.start_drag)
        self.root.bind('<B1-Motion>', self.drag_window)
        
        x_pos = (screen_width - self.overlay_width) // 2
        y_pos = screen_height - overlay_height - 20
        self.root.geometry(f"{self.overlay_width}x{overlay_height}+{x_pos}+{y_pos}")
        
        self._update_max_chars_per_line_estimate()
        self.root.after(self.silence_check_interval_ms, self._check_for_silence)
        self.root.mainloop()

    def create_overlay(self):
        if self.root is not None:
            return
        self.overlay_thread = threading.Thread(target=self._run_tkinter, daemon=True)
        self.overlay_thread.start()
        time.sleep(0.5)

    def start_drag(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def drag_window(self, event):
        x = self.root.winfo_x() + event.x - self.drag_start_x
        y = self.root.winfo_y() + event.y - self.drag_start_y
        self.root.geometry(f"+{x}+{y}")

    def show(self):
        def _show():
            if self.root is not None and not self.is_visible:
                self.root.deiconify()
                self.root.lift()
                self.is_visible = True
                self.last_update_time = time.time()
                print("Overlay shown")
        
        if self.root is None:
            self.create_overlay()
            time.sleep(0.5)
        
        if self.root is not None:
            self.root.after(0, _show)

    def hide(self):
        def _hide():
            if self.root is not None and self.is_visible:
                self.root.withdraw()
                self.is_visible = False
                print("Overlay hidden after long silence")
        
        if self.root:
            self.root.after(0, _hide)

    def set_font_size(self, size):
        if not isinstance(size, (int, float)) or size <= 0:
            return
        self.current_font = (self.font_family, int(size), self.font_weight)
        if self.label:
            self.label.config(font=self.current_font)
        if self.root:
            self._update_max_chars_per_line_estimate()

    def _update_max_chars_per_line_estimate(self):
        if self.label and self.root:
            test_font = tkFont.Font(font=self.current_font)
            test_string = "The quick brown fox jumps over the lazy dog."
            width_of_test_string = test_font.measure(test_string)
            effective_width = self.overlay_width - (self.label.cget('padx') * 2)
            if width_of_test_string > 0:
                avg_char_width = width_of_test_string / len(test_string)
                self.max_chars_per_line_estimate = int(effective_width / avg_char_width * 0.9)
            else:
                self.max_chars_per_line_estimate = 60
        else:
            self.max_chars_per_line_estimate = 60

    def _wrap_text_into_logical_lines(self, text):
        """Wrap text into lines based on the current width estimate"""
        if not text:
            return []
        
        words = text.split()
        if not words:
            return []
        
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_line and (current_length + word_length + 1 > self.max_chars_per_line_estimate):
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += (1 if current_line else 0) + word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else ['']

    def _check_for_silence(self):
        """Check if it's time to hide the captions after silence"""
        if self.is_visible and (time.time() - self.last_update_time) > self.hide_timeout_seconds:
            self.hide()
        
        if self.root:
            self.root.after(self.silence_check_interval_ms, self._check_for_silence)

    def _fix_character_encoding(self, text):
        """Fix common encoding issues in the text, including raw byte sequences."""
        if not text:
            return text
        
        def process_byte_sequences(text):
            def replace_hex_sequence(match):
                try:
                    hex_values = re.findall(r'<0x([0-9A-Fa-f]{2})>', match.group(0))
                    if not hex_values:
                        return match.group(0)
                    
                    byte_data = bytes(int(h, 16) for h in hex_values)
                    try:
                        return byte_data.decode('utf-8')
                    except UnicodeDecodeError:
                        return byte_data.decode('cp1251')
                except Exception as e:
                    print(f"Error decoding sequence {match.group(0)}: {e}")
                    return match.group(0)
            
            return re.sub(r'(?:<0x[0-9A-Fa-f]{2}>)+', replace_hex_sequence, text)
        
        text = process_byte_sequences(text)
        
        for wrong, right in self.character_corrections.items():
            text = text.replace(wrong, right)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def update_text(self, new_full_fragment, is_live_transcription=True):
        with self.text_lock:
            processed_fragment = self._fix_character_encoding(new_full_fragment.strip())
            self.hide_timeout_seconds = self.translation_hide_timeout if not is_live_transcription else self.normal_hide_timeout
            
            if not any(char.isalnum() for char in processed_fragment):
                return
            
            words = processed_fragment.split()
            processed_words = [self._fix_character_encoding(word) for word in words]
            processed_text = ' '.join(processed_words)
            processed_text = self._fix_character_encoding(processed_text)
            
            candidate_lines = self._wrap_text_into_logical_lines(processed_text)
            if not candidate_lines:
                return
            
            temp_display_state = collections.deque(candidate_lines[-2:], maxlen=2)
            potential_display_content = "\n".join(list(temp_display_state))
            
            if potential_display_content != self._last_displayed_content:
                if not self.is_visible:
                    self.show()
                    time.sleep(0.05)
                
                if not is_live_transcription:
                    self.showing_translation = True
                    current_time = time.time()
                    time_since_last = current_time - self.last_sentence_time
                    self.last_sentence_time = current_time
                    
                    if self.typewriter_job is not None and self.root:
                        self.root.after_cancel(self.typewriter_job)
                    
                    self.pending_sentences.append(potential_display_content)
                    
                    if self.typewriter_index >= len(self.typewriter_target_words) or time_since_last < self.sentence_gap_threshold:
                        self._process_next_sentence()
                else:
                    self.display_lines_queue = temp_display_state
                    self.text_buffer = potential_display_content
                    self._last_displayed_content = potential_display_content
                    self.last_update_time = time.time()
                    
                    if self.root and is_live_transcription:
                        self.root.after(0, self._update_display)

    def _update_display(self):
        if self.label:
            self.label.config(text=self.text_buffer)

    def _typewrite_text(self):
        """Handle the typewriter effect animation - word by word with scrolling behavior"""
        if self.typewriter_index < len(self.typewriter_target_words):
            self.typewriter_words = self.typewriter_target_words[:self.typewriter_index + 1]
            self.typewriter_index += 1
            
            full_text = ' '.join(self.typewriter_words)
            wrapped_lines = self._wrap_text_into_logical_lines(full_text)
            
            if len(wrapped_lines) > 2:
                display_lines = wrapped_lines[-2:]
            else:
                display_lines = wrapped_lines
            
            self.text_buffer = '\n'.join(display_lines)
            if self.root:
                self._update_display()
                self.typewriter_job = self.root.after(self.typewriter_speed, self._typewrite_text)
        else:
            self._process_next_sentence()

    def _process_next_sentence(self):
        """Process the next sentence in the queue if available"""
        current_time = time.time()
        
        if (current_time - self.last_sentence_time) < self.sentence_gap_threshold and self.pending_sentences:
            next_sentence = self.pending_sentences.pop(0)
            new_words = next_sentence.split()
            if new_words:
                if self.typewriter_target_words:
                    self.typewriter_target_words.append('')
                self.typewriter_target_words.extend(new_words)
                self._typewrite_text()
            return
        
        word_count = len(self.typewriter_target_words)
        if word_count > 0:
            display_time = max(2000, word_count * self.ms_per_word)
            if self.root:
                self.root.after(display_time, self._check_for_silence_after_display)

    def _check_for_silence_after_display(self):
        """Check if we should hide after the display time has elapsed"""
        if self.is_visible and (time.time() - self.last_update_time) > (self.hide_timeout_seconds / 2):
            self.hide()
            self.showing_translation = False

    def clear_text_only(self):
        """Clears the text content"""
        with self.text_lock:
            if self.typewriter_job is not None and self.root:
                self.root.after_cancel(self.typewriter_job)
            self.typewriter_job = None
            self.text_buffer = ""
            self.typewriter_words = []
            self.typewriter_target_words = []
            self.typewriter_index = 0
            self.pending_sentences = []
            self.display_lines_queue.clear()
            self._last_displayed_content = ""
            self.showing_translation = False
            self.last_sentence_time = time.time()
            if self.root:
                self.root.after(0, self._update_display)

    def destroy(self):
        if self.root is not None:
            self.root.quit()
            self.root.destroy()
            self.root = None
            self.label = None
            self.is_visible = False
