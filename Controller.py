import keyboard
import pyautogui
pyautogui.FAILSAFE = False

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

        # Different modes:
        # there are only 2 states: insert mode and normal mode
        # insert mode frees up the whole keyboard for typing, normal mode captures keys for cursor control
        self.in_insert_mode = True 
        self.current_keybinds = []


    def handle_move_press(self, key_name):
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

    # def handle_space(self, e):
    #     if not self.in_insert_mode:
    #         self.should_calibrate_point = True

    

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
        # self.current_keybinds.append(keyboard.on_press_key('space', self.handle_space, suppress=True))

        keyboard.on_press_key('s', lambda e: self.handle_move_press('s'), suppress=True)
        keyboard.on_release_key('s', lambda e: self.handle_move_release('s'), suppress=True)
        
        keyboard.on_press_key('f', lambda e: self.handle_move_press('f'), suppress=True)
        keyboard.on_release_key('f', lambda e: self.handle_move_release('f'), suppress=True)

        keyboard.on_press_key('e', lambda e: self.handle_move_press('e'), suppress=True)
        keyboard.on_release_key('e', lambda e: self.handle_move_release('e'), suppress=True)

        keyboard.on_press_key('d', lambda e: self.handle_move_press('d'), suppress=True)
        keyboard.on_release_key('d', lambda e: self.handle_move_release('d'), suppress=True)
        print("Normal mode activated. Keybindings are active.")
        

    
