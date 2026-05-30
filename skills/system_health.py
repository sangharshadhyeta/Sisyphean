# Sisyphean skill — report system resource usage (CPU, memory, disk, uptime)
import sys


def get_health() -> str:
    try:
        import psutil
        import platform
        import datetime

        cpu     = psutil.cpu_percent(interval=0.5)
        mem     = psutil.virtual_memory()
        disk    = psutil.disk_usage("/")
        boot_ts = datetime.datetime.fromtimestamp(psutil.boot_time())
        uptime  = datetime.datetime.now() - boot_ts

        hours, rem = divmod(int(uptime.total_seconds()), 3600)
        minutes    = rem // 60

        lines = [
            f"CPU:    {cpu:.1f}%",
            f"Memory: {mem.percent:.1f}%  ({mem.used // 1024**2} MB / {mem.total // 1024**2} MB)",
            f"Disk:   {disk.percent:.1f}%  ({disk.used // 1024**3} GB / {disk.total // 1024**3} GB)",
            f"Uptime: {hours}h {minutes}m",
            f"OS:     {platform.system()} {platform.release()}",
        ]
        return "\n".join(lines)

    except ImportError:
        # psutil not available — fall back to platform-only info
        import platform
        return f"OS: {platform.system()} {platform.release()} (install psutil for full health data)"

    except Exception as exc:
        return f"Error: {exc}"


def main():
    print(get_health())


if __name__ == "__main__":
    main()
