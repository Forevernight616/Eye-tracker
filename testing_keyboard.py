import keyboard
import time
import pyautogui
pyautogui.FAILSAFE = False

should_exit = False
should_reset_calibration = False
should_calibrate_point = False

# Global flags for continuous movement
is_moving_left = False
is_moving_right = False
is_moving_up = False
is_moving_down = False

# Different modes:
# there are only 2 states: insert mode and normal mode
# insert mode frees up the whole keyboard for typing, normal mode captures keys for cursor control
in_insert_mode = True 
current_keybinds = []


def handle_move_press(key_name):
    global is_moving_left, is_moving_right, is_moving_up, is_moving_down
    if key_name == 's':
        is_moving_left = True
    elif key_name == 'f':
        is_moving_right = True
    elif key_name == 'e':
        is_moving_up = True
    elif key_name == 'd':
        is_moving_down = True

# Release handlers set the flag back to False
def handle_move_release(key_name):
    global is_moving_left, is_moving_right, is_moving_up, is_moving_down
    if key_name == 's':
        is_moving_left = False
    elif key_name == 'f':
        is_moving_right = False
    elif key_name == 'e':
        is_moving_up = False
    elif key_name == 'd':
        is_moving_down = False


# Handler function
def handle_q(e):
    global should_exit
    if not in_insert_mode:
        should_exit = True

def handle_r(e):
    global should_reset_calibration
    if not in_insert_mode:
        should_reset_calibration = True

def handle_space(e):
    global should_calibrate_point
    if not in_insert_mode:
        should_calibrate_point = True

def enter_insert_mode():
    global in_insert_mode
    if in_insert_mode:
        return

    in_insert_mode = True
    print("Entering insert mode, removing keybindings.")

    # Unhook all movement and action keys using the stored hook objects
    keyboard.unhook_all()
    keyboard.on_press_key('`', lambda e: enter_normal_mode(), suppress=True) 
    current_keybinds.clear()
    
    print("Insert mode activated. Keyboard is free for typing.")

def enter_normal_mode():
    global in_insert_mode
    if not in_insert_mode:
        return

    in_insert_mode = False
    # Re-hook all movement and action keys
    print("Entering normal mode, setting up keybindings.")
    
    current_keybinds.append(keyboard.on_press_key('q', handle_q, suppress=True))
    current_keybinds.append(keyboard.on_press_key('r', handle_r, suppress=True))
    current_keybinds.append(keyboard.on_press_key('space', handle_space, suppress=True))

    keyboard.on_press_key('s', lambda e: handle_move_press('s'), suppress=True)
    keyboard.on_release_key('s', lambda e: handle_move_release('s'), suppress=True)
    
    keyboard.on_press_key('f', lambda e: handle_move_press('f'), suppress=True)
    keyboard.on_release_key('f', lambda e: handle_move_release('f'), suppress=True)
    
    keyboard.on_press_key('e', lambda e: handle_move_press('e'), suppress=True)
    keyboard.on_release_key('e', lambda e: handle_move_release('e'), suppress=True)
    
    keyboard.on_press_key('d', lambda e: handle_move_press('d'), suppress=True)
    keyboard.on_release_key('d', lambda e: handle_move_release('d'), suppress=True)
    print("Normal mode activated. Keybindings are active.")
    
def main():
    global should_exit
    global should_reset_calibration
    global should_calibrate_point
    global in_insert_mode

    keyboard.on_press_key('i', lambda e: enter_insert_mode(), suppress=False)
    keyboard.on_press_key('`', lambda e: enter_normal_mode(), suppress=True)
    MOVE_SPEED = 10  # pixels per movement step 
    
    enter_normal_mode()
    while True:
        if in_insert_mode:
            time.sleep(0.08)
            continue

        if should_exit:
            print("Exit flag detected. Exiting main loop.")
            break
        elif should_reset_calibration:
            print("Reset calibration flag detected. Resetting calibration.")
            should_reset_calibration = False  # Reset the flag after handling
        elif should_calibrate_point:
            print("Calibrate point flag detected. Calibrating point.")
            should_calibrate_point = False  # Reset the flag after handling

        x_change = 0
        y_change = 0

        if is_moving_left:
            x_change -= MOVE_SPEED
        if is_moving_right:
            x_change += MOVE_SPEED
        if is_moving_up:
            y_change -= MOVE_SPEED
        if is_moving_down:
            y_change += MOVE_SPEED
            
        # Only call pyautogui.move() if a movement key is pressed
        if x_change != 0 or y_change != 0:
            pyautogui.move(x_change, y_change, duration=0)
        
        time.sleep(0.01) # Sleep briefly to prevent busy-waiting

    keyboard.unhook_all()
    
if __name__ == "__main__":
    main()