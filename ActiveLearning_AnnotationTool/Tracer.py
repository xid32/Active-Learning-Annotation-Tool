
import inspect, os, socket, sys


class Tracer:
    current_frame = inspect.currentframe()

    debug_prefix = os.path.split(current_frame.f_code.co_filename)[0]

    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    cache = {}

    pid = os.getpid()

    def do_cache(self, file_path):
        with open(file_path) as file_handle:
            content = file_handle.read()
            lines = content.split('\n')
            for i, v in enumerate(lines):
                self.cache[f'{file_path}@{i + 1}'] = v

    def get_line(self, file_path, line):
        key = f'{file_path}@{line}'
        if key not in self.cache: self.do_cache(file_path)

        return self.cache[key]

    def on_line(self, frame, event, arg):
        pid = f'{self.pid}'.ljust(8, ' ')
        file_name = os.path.split(frame.f_code.co_filename)[1].ljust(32, ' ')
        line = self.get_line(frame.f_code.co_filename, frame.f_lineno)

        self.udp_socket.sendto(f'{pid} @ {file_name} -> {line}'.encode(), ('127.0.0.1', 10001))

    def trace_line(self, frame, event, arg):

        if event == 'line' and frame.f_code.co_filename.startswith(self.debug_prefix): self.on_line(frame, event, arg)

        return self.trace_line


    def trace_call(self, frame, event, arg):

        if event == 'call' and frame.f_code.co_filename.startswith(self.debug_prefix):
            return self.trace_line

        return None




tracer = Tracer()
sys.settrace(tracer.trace_call)

