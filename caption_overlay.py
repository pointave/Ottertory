import tkinter as tk
import threading
import time
import collections  # For deque
import tkinter.font as tkFont  # To measure font size
import re

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
        # Time before the overlay is completely hidden after silence
        self.hide_timeout_seconds = 2.0
        self.silence_check_interval_ms = 100  # More responsive checking
        
        # Character correction map for fixing encoding issues
        self.character_corrections = {
            # Spanish
            'Ã±': 'ñ',
            # Russian
            'Ð¶': 'ж',
            'Ñ\x86': 'ц',
            'Ñ\x8d': 'э',
            'Ñ\x85': 'х',
            'Ñ„': 'ф',
            'Ð°': 'а',
            'Ð±': 'б',
            'Ð²': 'в',
            'Ð³': 'г',
            'Ð´': 'д',
            'Ðµ': 'е',
            'Ð·': 'з',
            'Ð¸': 'и',
            'Ð¹': 'й',
            'Ðº': 'к',
            'Ð»': 'л',
            'Ð¼': 'м',
            'Ð½': 'н',
            'Ð¾': 'о',
            'Ð¿': 'п',
            'Ñ\x80': 'р',
            'Ñ\x81': 'с',
            'Ñ\x82': 'т',
            'Ñ\x83': 'у',
            'Ñ\x84': 'ф',
            'Ñ\x85': 'х',
            'Ñ\x86': 'ц',
            'Ñ\x87': 'ч',
            'Ñ\x88': 'ш',
            'Ñ\x89': 'щ',
            'Ñ\x8a': 'ъ',
            'Ñ\x8b': 'ы',
            'Ñ\x8c': 'ь',
            'Ñ\x8d': 'э',
            'Ñ\x8e': 'ю',
            'Ñ\x8f': 'я',
            'Ð•': 'Е',
            'Ð¡': 'С',
            'Ð£': 'У',
            'Ð¯': 'Я'
        }

    def _on_mouse_wheel(self, event):
        """Adjust window opacity based on mouse wheel movement"""
        current_alpha = float(self.root.attributes('-alpha'))
        if event.delta > 0:
            new_alpha = min(1.0, current_alpha + 0.05)
        else:
            new_alpha = max(0.05, current_alpha - 0.05)
        self.root.attributes('-alpha', new_alpha)

    def _run_tkinter(self):
        """Run the tkinter main loop in a separate thread"""
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Closed Captions")
        self.root.configure(bg='black')
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.7)
        self.root.overrideredirect(True)
        self.root.bind('<MouseWheel>', self._on_mouse_wheel)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.overlay_width = min(600, int(screen_width * 0.3))
        overlay_height = int(screen_height * 0.06)
        self.set_font_size(max(16, int(screen_height * 0.02)))

        self.frame = tk.Frame(self.root, bg='black')
        self.frame.pack(fill='both', expand=True)

        self.label = tk.Label(
            self.frame, text="", font=self.current_font, fg='white', bg='black',
            wraplength=self.overlay_width, anchor='w', justify='left',
            padx=10, pady=5, bd=0, highlightthickness=0
        )
        self.label.pack(side='left', fill='both', expand=True)

        self.root.bind('<Button-1>', self.start_drag)
        self.root.bind('<B1-Motion>', self.drag_window)

        x_pos = (screen_width - self.overlay_width) // 2
        y_pos = screen_height - overlay_height - 20
        self.root.geometry(f"{self.overlay_width}x{overlay_height}+{x_pos}+{y_pos}")

        self._update_max_chars_per_line_estimate()
        self.root.after(self.silence_check_interval_ms, self._check_for_silence)
        self.root.mainloop()

    def create_overlay(self):
        if self.root is not None: return
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
                self.last_update_time = time.time() # Reset timer to prevent immediate hide
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
        if not isinstance(size, (int, float)) or size <= 0: return
        self.current_font = ("Arial", int(size), "bold")
        if self.label:
            self.label.config(font=self.current_font)
            if self.root: self._update_max_chars_per_line_estimate()

    def _update_max_chars_per_line_estimate(self):
        # This helper function remains the same
        if self.label and self.root:
            test_font = tkFont.Font(font=self.current_font)
            test_string = "The quick brown fox jumps over the lazy dog."
            width_of_test_string = test_font.measure(test_string)
            effective_width = self.overlay_width - (self.label.cget('padx') * 2)
            if width_of_test_string > 0:
                avg_char_width = width_of_test_string / len(test_string)
                self.max_chars_per_line_estimate = int(effective_width / avg_char_width * 0.9)
            else: self.max_chars_per_line_estimate = 60
        else: self.max_chars_per_line_estimate = 60

    def _wrap_text_into_logical_lines(self, text):
        # This helper function remains the same
        if not text: return []
        words = text.split()
        lines, current_line = [], ""
        for word in words:
            if len(current_line) + len(word) + 1 > self.max_chars_per_line_estimate:
                lines.append(current_line)
                current_line = word
            else: current_line += (" " if current_line else "") + word
        if current_line: lines.append(current_line)
        return lines

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

        # First, try to fix raw byte sequences like <0xD0><0xAD>
        def process_byte_sequences(text):
            # Handle sequences like <0xD0><0xAD> or <0xd0><0xad>
            def replace_hex_sequence(match):
                try:
                    # Extract all hex values from the match
                    hex_values = re.findall(r'<0x([0-9A-Fa-f]{2})>', match.group(0))
                    if not hex_values:
                        return match.group(0)
                        
                    # Convert hex strings to bytes
                    byte_data = bytes(int(h, 16) for h in hex_values)
                    
                    # Try to decode as UTF-8 first, then fall back to cp1251 for Russian
                    try:
                        return byte_data.decode('utf-8')
                    except UnicodeDecodeError:
                        return byte_data.decode('cp1251')
                        
                except Exception as e:
                    print(f"Error decoding sequence {match.group(0)}: {e}")
                    return match.group(0)
            
            # Replace all occurrences of hex sequences
            return re.sub(r'(?:<0x[0-9A-Fa-f]{2}>)+', replace_hex_sequence, text)
        
        # Process byte sequences first
        text = process_byte_sequences(text)
        
        # Then fix any remaining common encoding issues
        for wrong, right in self.character_corrections.items():
            text = text.replace(wrong, right)
            
        # Clean up any remaining artifacts
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def update_text(self, new_full_fragment, is_live_transcription=False):
        with self.text_lock:
            # First fix encoding of the raw input
            processed_fragment = self._fix_character_encoding(new_full_fragment.strip())

            if not any(char.isalnum() for char in processed_fragment):
                return

            # Split into words and process each word's encoding
            words = processed_fragment.split()
            processed_words = [self._fix_character_encoding(word) for word in words]
            
            # Rebuild the text with fixed words
            processed_text = ' '.join(processed_words)
            
            # Process the final text again to catch any multi-word patterns
            processed_text = self._fix_character_encoding(processed_text)
            
            # Generate lines from the fully processed text
            candidate_lines = self._wrap_text_into_logical_lines(processed_text)
            if not candidate_lines: 
                return

            temp_display_state = collections.deque(candidate_lines[-2:], maxlen=2)
            potential_display_content = "\n".join(list(temp_display_state))

            if potential_display_content != self._last_displayed_content:
                if not self.is_visible:
                    self.show()
                    time.sleep(0.05)
                    
                self.display_lines_queue = temp_display_state
                self.text_buffer = potential_display_content
                self._last_displayed_content = potential_display_content
                self.last_update_time = time.time()  # Reset silence timer on new text
                if self.root:
                    self.root.after(0, self._update_display)

    def _update_display(self):
        if self.label:
            self.label.config(text=self.text_buffer)

    def clear_text_only(self):
        """Clears the text content"""
        with self.text_lock:
            self.text_buffer = ""
            self.display_lines_queue.clear()
            self._last_displayed_content = ""
            if self.root:
                self.root.after(0, self._update_display)

    def destroy(self):
        if self.root is not None:
            self.root.quit()
            self.root.destroy()
            self.root = None
            self.label = None
            self.is_visible = False
