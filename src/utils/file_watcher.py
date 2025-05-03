import os
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class PythonFileHandler(FileSystemEventHandler):
    def __init__(self, app_process=None):
        self.app_process = app_process
        self.restart_needed = False

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.py'):
            logging.info(f"Python file modified: {event.src_path}")
            self.restart_needed = True

    def restart_app(self):
        if self.restart_needed and self.app_process:
            logging.info("Restarting application...")
            self.app_process.terminate()
            self.app_process.wait()
            main_script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src', 'main.py')
            env = os.environ.copy()
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = project_root
            self.app_process = subprocess.Popen([sys.executable, main_script], env=env, cwd=project_root)
            self.restart_needed = False
        return self.app_process

def main():
    # Get project root and src directories
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    src_path = os.path.join(project_root, 'src')
    
    # Set up Python path for proper module resolution
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root
        
    event_handler = PythonFileHandler()
    observer = Observer()
    observer.schedule(event_handler, src_path, recursive=True)
    observer.start()

    main_script = os.path.join(src_path, 'main.py')
    app_process = subprocess.Popen([sys.executable, main_script], env=env, cwd=project_root)
    event_handler.app_process = app_process

    try:
        while True:
            time.sleep(1)
            app_process = event_handler.restart_app()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        if app_process:
            app_process.terminate()
            app_process.wait()
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()