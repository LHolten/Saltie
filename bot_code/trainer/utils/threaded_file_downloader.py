import collections
import queue
import threading


class ThreadedFileDownloader:
    def __init__(self, max_files, num_downloader_threads, num_trainer_threads,
                 get_file_list_function, file_getter_function, file_processor_function, batches):
        self.downloaded_files = queue.Queue(maxsize=max_files * batches) # this is in single file form
        self.files_to_download = queue.Queue(maxsize=batches) # this is in string form, so each string is one batch
        self.processed_counter = 0
        self.max_files = max_files
        self.num_downloader_threads = num_downloader_threads
        self.num_trainer_threads = num_trainer_threads
        self.process_file = file_processor_function
        self.file_getter_function = file_getter_function
        self.get_file_list_function = get_file_list_function
        self.batches = batches
        self.counter = 0
        self.total_time = 0
        self.total_files = 1

    def file_processor_worker(self):
        while True:
            file = self.downloaded_files.get()
            if file is None and self.files_to_download.empty() and self.downloaded_files.empty():
                break
            if file is None:
                continue
            print('running file', self.counter + 1, '/', self.counter + self.downloaded_files.qsize() + 1)
            self.total_time += self.process_file(file)
            self.counter += 1
            self.downloaded_files.task_done()

    def downloader_worker(self):
        while True:
            file_name = self.files_to_download.get()
            if file_name is None:
                break
            try:
                files = self.file_getter_function(file_name)
                if not self.is_sequence(files):
                    files = [files]
                for f in files:
                    self.downloaded_files.put(f)
            except ConnectionError as e:
                print(e)
            self.files_to_download.task_done()

    def get_replay_list(self):
        files = self.get_file_list_function(self.max_files, False, self.batches)
        print("Downloading", str(files).replace(';', '\n'))
        self.total_files = len(files)
        print('training on ' + str(self.total_files * self.batches) + ' files')
        for replay in files:
            self.files_to_download.put(replay)

    def create_and_run_workers(self):
        print('creating', self.num_downloader_threads, 'downloader threads')
        for i in range(self.num_downloader_threads):
            t = threading.Thread(target=self.downloader_worker)
            t.daemon = True
            t.start()

        print('creating', self.num_trainer_threads, 'trainer threads')
        for i in range(self.num_trainer_threads):
            t = threading.Thread(target=self.file_processor_worker)
            t.daemon = True
            t.start()

        self.get_replay_list()

        # this order is important
        self.files_to_download.join()
        self.downloaded_files.join()

        print('ran through all files in ' + str(self.total_time / 60) + ' minutes')
        print('ran through all files in ' + str(self.total_time / 3600) + ' hours')
        print('average time per file: ' + str((self.total_time / max(1, self.total_files))) + ' seconds')

    @staticmethod
    def is_sequence(obj):
        if isinstance(obj, str):
            return False
        return isinstance(obj, collections.Sequence)
