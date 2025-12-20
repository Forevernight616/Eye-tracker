import keyboard
import pyautogui
import time
pyautogui.FAILSAFE = False

MANUAL_MOVE_COOLDOWN = 1.5
SCROLL_AMOUNT = 55  # number of pixel per scroll

class Controller:
    def __init__(self):
        self.should_exit = False
        self.should_reset_calibration = False
        self.should_calibrate_point = False

        # Global flags for continuous movement
        self.is_moving_left = False
        self.is_moving_right = False
        self.is_moving_up = False
        self.is_moving_down = False

        # track last manual movement time 
        self.last_manual_move_time = 0

        # flags for scrolling
        self.is_scrolling_down = False
        self.is_scrolling_up = False

        # Different modes:
        self.in_insert_mode = True #start in insert mode
        self.current_keybinds = []

    def getScrollAmount(self):
        return SCROLL_AMOUNT

    def is_manual_movement_active(self):
        return (self.is_moving_left or self.is_moving_right or 
                self.is_moving_up or self.is_moving_down)

    def should_use_eye_tracking(self):
        if self.in_insert_mode:
            return False    
        if self.is_manual_movement_active():
            return False
        
        # Check if enough time has passed since the last manual movement
        time_since_last_move = time.time() - self.last_manual_move_time
        return time_since_last_move > MANUAL_MOVE_COOLDOWN

    def handle_move_press(self, key_name):
        self.last_manual_move_time = time.time()
        if key_name == 's':
            self.is_moving_left = True
        elif key_name == 'f':
            self.is_moving_right = True
        elif key_name == 'e':
            self.is_moving_up = True
        elif key_name == 'd':
            self.is_moving_down = True

    # Release handlers set the flag back to False
    def handle_move_release(self, key_name):
        self.last_manual_move_time = time.time()
        if key_name == 's':
            self.is_moving_left = False
        elif key_name == 'f':
            self.is_moving_right = False
        elif key_name == 'e':
            self.is_moving_up = False
        elif key_name == 'd':
            self.is_moving_down = False

    # Handler function
    def handle_q(self,e):
        if not self.in_insert_mode:
            self.should_exit = True

    def handle_r(self,e):
        if not self.in_insert_mode:
            self.should_reset_calibration = True

    def right_click(self, e):
        if not self.in_insert_mode:
            pyautogui.rightClick()


    def press_down_left_click(self):
        if not self.in_insert_mode:
            pyautogui.mouseDown(button='left')
    def release_left_click(self):
        if not self.in_insert_mode:
            pyautogui.mouseUp(button='left')

    def handle_scroll_press(self, key_name):
        self.last_manual_move_time = time.time()
        if key_name == 'l':
            self.is_scrolling_down = True
        elif key_name == ';':
            self.is_scrolling_up = True

    def handle_scroll_release(self, key_name):
        self.last_manual_move_time = time.time()
        if key_name == 'l':
            self.is_scrolling_down = False
        elif key_name == ';':
            self.is_scrolling_up = False

    def enter_insert_mode(self):
        if self.in_insert_mode:
            return

        self.in_insert_mode = True
        print("Entering insert mode, removing keybindings.")

        # Unhook all movement and action keys using the stored hook objects
        keyboard.unhook_all()
        keyboard.on_press_key('`', lambda e: self.enter_normal_mode(), suppress=True) 
        self.current_keybinds.clear()
        
        print("Insert mode activated. Keyboard is free for typing.")

    def enter_normal_mode(self):        
        if not self.in_insert_mode:
            return

        self.in_insert_mode = False
        # Re-hook all movement and action keys
        print("Entering normal mode, setting up keybindings.")
        
        self.current_keybinds.append(keyboard.on_press_key('q', self.handle_q, suppress=True))
        self.current_keybinds.append(keyboard.on_press_key('r', self.handle_r, suppress=True))
        self.current_keybinds.append(keyboard.on_press_key('k', self.right_click, suppress=True))

        keyboard.on_press_key('j', lambda e: self.press_down_left_click(), suppress=True)
        keyboard.on_release_key('j', lambda e: self.release_left_click(), suppress=True)

        keyboard.on_press_key('s', lambda e: self.handle_move_press('s'), suppress=True)
        keyboard.on_release_key('s', lambda e: self.handle_move_release('s'), suppress=True)
        
        keyboard.on_press_key('f', lambda e: self.handle_move_press('f'), suppress=True)
        keyboard.on_release_key('f', lambda e: self.handle_move_release('f'), suppress=True)

        keyboard.on_press_key('e', lambda e: self.handle_move_press('e'), suppress=True)
        keyboard.on_release_key('e', lambda e: self.handle_move_release('e'), suppress=True)

        keyboard.on_press_key('d', lambda e: self.handle_move_press('d'), suppress=True)
        keyboard.on_release_key('d', lambda e: self.handle_move_release('d'), suppress=True)

        keyboard.on_press_key('l', lambda e: self.handle_scroll_press('l'), suppress=True)
        keyboard.on_release_key('l', lambda e: self.handle_scroll_release('l'), suppress=True)

        keyboard.on_press_key(';', lambda e: self.handle_scroll_press(';'), suppress=True)
        keyboard.on_release_key(';', lambda e: self.handle_scroll_release(';'), suppress=True)
        
        print("Normal mode activated. Keybindings are active.")
        

    
