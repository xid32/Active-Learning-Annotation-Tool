
import sys

o_stdout = sys.stdout
o_stderr = sys.stderr

class DummyOut:

    file_out = None

    def __init__(self):
        self.file_out = open('output.txt', 'w')

    def write(self, content):
        o_stdout.write(content)
        self.file_out.write(content)
    
    def flush(self):
        o_stdout.flush()
        self.file_out.flush()


dummy_out = DummyOut()

sys.stdout = dummy_out
sys.stderr = dummy_out
