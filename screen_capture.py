import mss
import numpy as np
from PIL import Image

class ScreenCapturer:
    def __init__(self, bounding_box=None):
        """
        Initialize the screen capturer.
        
        Args:
            bounding_box (dict): Dictionary with 'top', 'left', 'width', 'height' keys.
                                Defaults to primary monitor size.
                                Note: This may need to be adjusted by the user for their specific setup.
        """
        if bounding_box is None:
            # Get actual screen dimensions
            try:
                with mss.mss() as sct:
                    monitor = sct.monitors[1]  # Primary monitor
                    self.bounding_box = {
                        'top': monitor['top'],
                        'left': monitor['left'],
                        'width': monitor['width'],
                        'height': monitor['height']
                    }
            except:
                # Fallback to 1920x1080 if we can't get screen info
                self.bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        else:
            self.bounding_box = bounding_box
            
        # Don't initialize mss here - we'll create it per capture to avoid threading issues
        self.sct = None
        
    def capture_frame(self):
        """
        Capture a screenshot of the defined bounding box.
        
        Returns:
            PIL.Image: RGB image of the captured screen region
        """
        try:
            # Create a new mss instance for each capture to avoid threading issues
            with mss.mss() as sct:
                # Grab screenshot using mss
                sct_img = sct.grab(self.bounding_box)
                
                # Convert BGRA data to RGB PIL Image
                # The correct conversion: Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
                pil_image = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
                
                return pil_image
                
        except Exception as e:
            # If mss fails, try alternative method using PIL
            try:
                import pyautogui
                # Capture screen using pyautogui as fallback
                screenshot = pyautogui.screenshot(region=(
                    self.bounding_box['left'], 
                    self.bounding_box['top'], 
                    self.bounding_box['width'], 
                    self.bounding_box['height']
                ))
                return screenshot
            except ImportError:
                # If pyautogui is not available, create a blank image
                blank_image = Image.new('RGB', (self.bounding_box['width'], self.bounding_box['height']), color='black')
                return blank_image
                
    def get_screen_info(self):
        """Get information about available monitors"""
        try:
            with mss.mss() as sct:
                return sct.monitors
        except Exception as e:
            return f"Error getting screen info: {str(e)}"

if __name__ == "__main__":
    # Demonstrate the class usage
    print("Testing ScreenCapturer...")
    
    # Create capturer instance
    capturer = ScreenCapturer()
    
    # Print screen info
    print("Screen info:")
    print(capturer.get_screen_info())
    print(f"Using bounding box: {capturer.bounding_box}")
    
    # Capture a frame
    print("Capturing screen...")
    image = capturer.capture_frame()
    
    # Save the captured image
    output_file = "test_capture.png"
    image.save(output_file)
    print(f"Screen capture saved to: {output_file}")
    print(f"Image size: {image.size}")
