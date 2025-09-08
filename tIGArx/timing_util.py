from mpi4py import MPI

from tigarx.common import mpirank


class TimingLogger:
    """
    A simple class to log the time taken by different parts of
    the code. It is intended to be used as a singleton that
    keeps track of the start of all execution times and prints
    them, if desired, as the program executes. When the program
    is done, the user can print the log to see the times taken
    for
    """
    def __init__(self):
        """
        Initializes the logger with empty lists for the start
        times, pending timings, and finished timings. A list of
        labels that correspond to each entry is also initialized.
        """
        self.start_times = []
        self.pending = []
        self.finished = []

        self.current_level = 0

        self.id_to_label = []
        self.id_counter = 0

    class Timer:
        """
        A simple class to measure the time taken by a specific
        part of the code. The timer is started when the object is
        created and stopped when the object is deleted. The user
        can also manually stop the timer by calling the
        `stop_timer` method on the timer object. Intended to be
        used instead of labels or with the 'with' statement.
        """
        def __init__(self, logger, label, print_on_start=False, print_on_end=True):
            self.logger = logger
            self.label = label
            self.print_on_start = print_on_start
            self.print_on_end = print_on_end

            self.logger.start_timing(self.label, self.print_on_start)

        def __enter__(self):
            """
            Do nothing because the timer starts at creation.
            """
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Stop the timer when the 'with' block ends.
            """
            self.stop_timer()

        def __del__(self):
            """
            Stop the timer when the object is deleted or goes out
            of scope.
            """
            self.stop_timer()

        def stop_timer(self):
            self.logger.end_timing(self.label, self.print_on_end)

        def __str__(self):
            return f"Timer('{self.label}') of {self.logger}"

    def create_timer(self, label, print_on_start=False, print_on_end=True):
        """
        Creates a timer object that is used to measure a specific
        part of the code. The timer is started when the object is
        created and stopped when it is deleted. The user can also
        manually stop the timer by calling the `stop_timer` method
        on the timer object. Also useful when using the 'with'
        statement.
        """
        return self.Timer(self, label, print_on_start, print_on_end)

    def start_timing(self, label, print=False):
        if mpirank == 0:
            log_id = self.id_counter
            self.id_to_label.append(label)
            self.id_counter += 1

            self.pending.append(log_id)
            self.start_times.append((log_id, MPI.Wtime()))
            self.current_level += 1

            if print:
                self.print_pending(self.pending[-1])

    def end_timing(self, label, print=True):
        if mpirank == 0:
            log_id = self.pending.pop()
            _label = self.id_to_label[log_id]

            if _label != label:
                raise ValueError(f"Timing for '{label}' was not started.")

            end_time = MPI.Wtime()
            _, start_time = self.start_times.pop()
            elapsed_time = end_time - start_time

            self.finished.append((log_id, self.current_level, elapsed_time))
            if print:
                self.print_completed(self.finished[-1])

            self.current_level -= 1

    def print_pending(self, log_id):
        label = self.id_to_label[log_id]
        self.print_start(label, self.current_level)

    def print_log(self):
        if mpirank == 0:
            for entry in self.finished:
                self.print_completed(entry)
            for log_id in self.pending:
                self.print_pending(log_id)

    def print_completed(self, entry):
        log_id, level, elapsed_time = entry
        label = self.id_to_label[log_id]
        self.print_end(label, level, elapsed_time)

    @staticmethod
    def print_start(label, level):
        indent = " " * (level * 4)
        print(f"{indent}{label}...")

    @staticmethod
    def print_end(label, level, elapsed_time):
        indent = " " * (level * 4)
        separation = " " * (max(0, 32 - len(label)))
        print(f"{indent}{label}:{separation} {elapsed_time:.3e} s")


perf_log = TimingLogger()
