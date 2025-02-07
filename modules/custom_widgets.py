from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QComboBox, QSizePolicy, QStyledItemDelegate)
from PyQt6.QtCore import Qt
from modules.utils import load_user_data, save_user_data


# Custom address menu where users can type in or select addresses
class AddressMenu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.user_data = load_user_data()  # Load stored addresses
        self.init_ui()

    def init_ui(self):
        # Main layout for the AddressMenu
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # ComboBox for selecting addresses
        self.combo_box = QComboBox()
        self.combo_box.setEditable(True)  # Allow typing in the combo box
        self.combo_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)  # Prevent auto-insert
        self.combo_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_box.addItems(self.user_data.get("last_folders", []))  # Load initial addresses
        self.combo_box.setCurrentText(self.user_data.get("last_folder", ""))  # Set current address
        layout.addWidget(self.combo_box)

        # Assign custom delegate to handle remove buttons
        self.combo_box.setItemDelegate(RemoveButtonDelegate(self.combo_box, self.remove_item))

    def remove_item(self, index):
        """Remove an address from the combo box and update user data."""
        if 0 <= index < self.combo_box.count():
            self.combo_box.removeItem(index)
            self.save_addresses()  # Update stored addresses

    def save_addresses(self):
        """Save the current list of addresses to user data, ensuring empty lists are handled."""
        addresses = [self.combo_box.itemText(i) for i in range(self.combo_box.count())]
        save_user_data(last_folders=addresses,
                       last_folder=self.combo_box.currentText().strip())  # Persist updated addresses

    def get_current_address(self):
        """Get the currently selected or typed address."""
        return self.combo_box.currentText().strip()

    def set_current_address(self, address):
        """Set the current address in the combo box."""
        self.combo_box.setCurrentText(address)

    def add_address(self, address):
        """Add a new address to the combo box if it doesn’t already exist."""
        if address and self.combo_box.findText(address) == -1:  # Avoid duplicates
            self.combo_box.insertItem(0, address)  # Add to the top
            self.combo_box.setCurrentIndex(0)  # Set as current
            self.save_addresses()  # Update user data

# Custom delegate to add a button inside each combo box item
class RemoveButtonDelegate(QStyledItemDelegate):
    def __init__(self, parent, remove_callback):
        super().__init__(parent)
        self.remove_callback = remove_callback  # Function to remove items

    def paint(self, painter, option, index):
        """Draw the item text and a small remove button."""
        super().paint(painter, option, index)
        painter.save()

        # Define button area (small red cross on the right)
        button_rect = option.rect.adjusted(option.rect.width() - 20, 2, -2, -2)
        painter.setPen(Qt.GlobalColor.red)
        painter.drawText(button_rect, Qt.AlignmentFlag.AlignCenter, "×")

        painter.restore()

    def editorEvent(self, event, model, option, index):
        """Handle button clicks inside the combo box."""
        if event.type() == event.Type.MouseButtonPress:
            button_rect = option.rect.adjusted(option.rect.width() - 20, 2, -2, -2)
            if button_rect.contains(event.pos()):
                self.remove_callback(index.row())  # Call the remove function
                return True
        return False