"""
run_demo.py  —  Entry point for all tasks

Usage
-----
  python run_demo.py               # runs direction recognition (default)
  python run_demo.py --task direction
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

def main():
    task = 'direction'
    if '--task' in sys.argv:
        idx  = sys.argv.index('--task')
        task = sys.argv[idx + 1]

    if task == 'direction':
        from tasks.direction_recognition import run_direction_task
        cfg  = os.path.join('config', 'direction_task.json')
        out  = os.path.join(os.path.dirname(__file__), 'direction_recognition.png')
        acc  = run_direction_task(cfg, out)
        print(f"\nFinal test accuracy: {acc*100:.1f}%")
    else:
        print(f"Unknown task '{task}'. Available: direction")

if __name__ == '__main__':
    main()