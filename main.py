#!/usr/bin/env python3
"""
Main entry point for the OLM Observer application.
Connects the engine and GUI components.
"""

from gui import App
from engine import Engine

def main():
    """Main function to start the application"""
    print("Starting OLM Observer...")
    
    # Create engine instance (initially with no GUI reference)
    engine = Engine()
    
    # Create GUI instance with engine reference
    app = App(engine)
    
    # Connect GUI back to engine
    engine.gui = app
    
    # Demonstrate different log types
    app.log_action("Application initialized successfully")
    app.log_system("GUI components loaded")
    print("Console output will appear in the Console Log panel")
    
    # Start the application
    app.run()

if __name__ == "__main__":
    main()
