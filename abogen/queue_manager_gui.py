# a simple window with a list of items in the queue, no checkboxes
# button to remove an item from the queue
# button to clear the queue

from PyQt5.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QDialogButtonBox,
    QLabel,
    QCheckBox,
    QSlider,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QListWidget,
)


class QueueManager(QDialog):
    def __init__(self, parent, queue: list, title="Queue Manager", size=(400, 300)):
        super().__init__()
        self.queue = queue
        self.parent = parent
        layout = QVBoxLayout()
        # list of queued items
        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode(QListWidget.SingleSelection)
        self.listwidget.setAlternatingRowColors(True)
        # add queue items to the list
        self.process_queue()

        layout.addWidget(self.listwidget)
        # add remove item button
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_item)
        layout.addWidget(remove_button)
        # add clear queue button
        clear_button = QPushButton("Clear Queue")
        clear_button.clicked.connect(self.clear_queue)
        layout.addWidget(clear_button)


        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self.setLayout(layout)

        self.setWindowTitle(title)
        self.setMinimumSize(*size)

    def process_queue(self):
        """Process the queue items."""
        if not self.queue:
            return
        for item in self.queue:
            # TODO take just the file name from the item, no directories
            self.listwidget.addItem(item.file_name)

    def remove_item(self):
        item = self.listwidget.currentItem()
        if not item:
            return
        found_item = [q for q in self.queue if q.file_name==item.text()][0]
        self.queue.remove(found_item)
        # rebind the queue to the listwidget
        self.listwidget.clear()
        self.process_queue()
        pass

    def clear_queue(self):
        self.queue.clear()
        self.listwidget.clear()

    def get_queue(self):
        return self.queue