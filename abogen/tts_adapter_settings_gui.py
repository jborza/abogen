"""
TTS Adapter Settings UI

This module provides GUI components for selecting and configuring TTS adapters.
"""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QTabWidget,
    QWidget,
    QMessageBox,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from abogen.tts_adapters.registry import TTSAdapterRegistry
from abogen.tts_settings import get_tts_config_manager, TTSConfig
from abogen.voice_manager import VoiceManager


class TTSAdapterSettingsDialog(QDialog):
    """Dialog for selecting and configuring TTS adapters."""
    
    # Signal emitted when adapter settings are applied
    adapter_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TTS Adapter Settings")
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        
        self.registry = TTSAdapterRegistry.get_instance()
        self.config_manager = get_tts_config_manager()
        self.tts_config = self.config_manager.get_configuration()
        
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("TTS Adapter Configuration")
        title_font = title.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Adapter selection
        adapter_layout = QHBoxLayout()
        adapter_layout.addWidget(QLabel("Active Adapter:"))
        
        self.adapter_combo = QComboBox()
        available_adapter_ids = self.registry.get_available_adapters()
        adapter_names = []
        self.adapter_id_map = {}  # Map display names to adapter IDs
        
        for adapter_id in available_adapter_ids:
            adapter_info = self.registry.get_adapter_info(adapter_id)
            if adapter_info:
                name = adapter_info.get('name', adapter_id)
                adapter_names.append(name)
                self.adapter_id_map[name] = adapter_id
        
        self.adapter_combo.addItems(adapter_names)
        
        # Set current adapter
        current_adapter = self.tts_config.active_adapter
        current_adapter_info = self.registry.get_adapter_info(current_adapter)
        if current_adapter_info:
            current_name = current_adapter_info.get('name', current_adapter)
            index = self.adapter_combo.findText(current_name)
            if index >= 0:
                self.adapter_combo.setCurrentIndex(index)
        
        self.adapter_combo.currentIndexChanged.connect(self.on_adapter_changed)
        adapter_layout.addWidget(self.adapter_combo)
        adapter_layout.addStretch()
        layout.addLayout(adapter_layout)

        # Adapter status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel()
        self.update_adapter_status()
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Tabs for adapter-specific settings
        self.tabs = QTabWidget()
        
        # Common settings tab
        common_tab = self.create_common_settings_tab()
        self.tabs.addTab(common_tab, "Common Settings")
        
        # Adapter-specific settings tab
        adapter_specific_tab = self.create_adapter_specific_tab()
        self.tabs.addTab(adapter_specific_tab, "Adapter Settings")
        
        layout.addWidget(self.tabs)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        button_layout.addWidget(apply_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_common_settings_tab(self):
        """Create the common settings tab."""
        widget = QWidget()
        layout = QFormLayout()

        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(['auto', 'cpu', 'cuda', 'mps'])
        
        device_config = self.tts_config.adapter_configs.get(
            self.tts_config.active_adapter, {}
        )
        current_device = device_config.get('device', 'auto')
        index = self.device_combo.findText(current_device)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)
        
        layout.addRow("Device:", self.device_combo)

        widget.setLayout(layout)
        return widget

    def create_adapter_specific_tab(self):
        """Create adapter-specific settings tab."""
        self.adapter_settings_widget = QWidget()
        self.adapter_settings_layout = QFormLayout()
        
        self.update_adapter_specific_settings()
        
        self.adapter_settings_widget.setLayout(self.adapter_settings_layout)
        return self.adapter_settings_widget

    def update_adapter_specific_settings(self):
        """Update adapter-specific settings based on selected adapter."""
        # Clear existing widgets
        while self.adapter_settings_layout.count() > 0:
            self.adapter_settings_layout.removeRow(0)

        # Get current adapter
        current_adapter_name = self.adapter_combo.currentText()
        current_adapter_id = self.adapter_id_map.get(current_adapter_name)
        
        if not current_adapter_id:
            return
        
        adapter_info = self.registry.get_adapter_info(current_adapter_id)
        if not adapter_info:
            return
        
        schema = adapter_info.get('configuration_schema', {})
        required_fields = schema.get('required_fields', {})
        optional_fields = schema.get('optional_fields', {})

        # Add description
        description = schema.get('description', '')
        if description:
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            self.adapter_settings_layout.addRow(desc_label)

        # Add required fields
        self.adapter_field_widgets = {}
        
        if required_fields:
            # Add required fields section
            required_label = QLabel("Required Fields:")
            required_font = required_label.font()
            required_font.setBold(True)
            required_label.setFont(required_font)
            self.adapter_settings_layout.addRow(required_label)
            
            for field_name, field_config in required_fields.items():
                field_type = field_config.get('type', 'string')
                default_value = field_config.get('default', '')
                description = field_config.get('description', '')
                
                # Create widget based on field type
                if field_type == 'boolean':
                    widget = QCheckBox(description)
                    widget.setChecked(default_value == True or default_value == 'true')
                else:
                    widget = QLineEdit()
                    widget.setPlaceholderText(description)
                    widget.setText(str(default_value))
                    widget.setToolTip(description)
                
                self.adapter_field_widgets[field_name] = widget
                
                if field_type != 'boolean':
                    self.adapter_settings_layout.addRow(field_name + " *", widget)
                else:
                    self.adapter_settings_layout.addRow(widget)
        
        # Add optional fields
        if optional_fields:
            # Add optional fields section
            optional_label = QLabel("Optional Fields:")
            optional_font = optional_label.font()
            optional_font.setBold(True)
            optional_label.setFont(optional_font)
            self.adapter_settings_layout.addRow(optional_label)
            
            for field_name, field_config in optional_fields.items():
                field_type = field_config.get('type', 'string')
                default_value = field_config.get('default', '')
                description = field_config.get('description', '')
                
                # Create widget based on field type
                if field_type == 'boolean':
                    widget = QCheckBox(description)
                    widget.setChecked(default_value == True or default_value == 'true')
                else:
                    widget = QLineEdit()
                    widget.setText(str(default_value))
                    widget.setToolTip(description)
                
                self.adapter_field_widgets[field_name] = widget
                
                if field_type != 'boolean':
                    self.adapter_settings_layout.addRow(field_name, widget)
                else:
                    self.adapter_settings_layout.addRow(widget)

    def on_adapter_changed(self, index):
        """Handle adapter selection change."""
        self.update_adapter_specific_settings()
        self.update_adapter_status()

    def update_adapter_status(self):
        """Update the adapter status display."""
        try:
            current_adapter_name = self.adapter_combo.currentText()
            current_adapter_id = self.adapter_id_map.get(current_adapter_name)
            
            if not current_adapter_id:
                self.status_label.setText("Unknown")
                return
            
            adapter_info = self.registry.get_adapter_info(current_adapter_id)
            if not adapter_info:
                self.status_label.setText("Unknown")
                return
            
            adapter_type = adapter_info.get('type', 'unknown')
            
            # Try to get adapter
            try:
                adapter = self.registry.get_adapter(current_adapter_id)
                status = f"{adapter_type.title()} - Initialized (Voices: {len(adapter.get_voices())})"
            except Exception as e:
                status = f"{adapter_type.title()} - Not initialized"
            
            self.status_label.setText(status)
            self.status_label.setStyleSheet("color: green;")
        except Exception as e:
            self.status_label.setText("Error")
            self.status_label.setStyleSheet("color: red;")

    def apply_settings(self):
        """Apply the selected settings."""
        try:
            # Get selected adapter
            current_adapter_name = self.adapter_combo.currentText()
            adapter_id = self.adapter_id_map.get(current_adapter_name)
            
            if not adapter_id:
                QMessageBox.warning(self, "Error", "Please select a valid adapter")
                return
            
            adapter_info = self.registry.get_adapter_info(adapter_id)
            if not adapter_info:
                QMessageBox.warning(self, "Error", "Please select a valid adapter")
                return
            
            # Update configuration
            self.tts_config.active_adapter = adapter_id
            
            # Update device setting
            device = self.device_combo.currentText()
            if adapter_id not in self.tts_config.adapter_configs:
                self.tts_config.adapter_configs[adapter_id] = {}
            self.tts_config.adapter_configs[adapter_id]['device'] = device
            
            # Update adapter-specific settings
            for field_name, widget in self.adapter_field_widgets.items():
                if isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                else:
                    value = widget.text()
                self.tts_config.adapter_configs[adapter_id][field_name] = value
            
            # Save configuration
            self.config_manager.save_configuration(self.tts_config)
            
            # Clear adapter cache to force reinitialization
            self.registry.clear_instances()
            
            # Emit signal to notify main GUI to reload voices
            self.adapter_changed.emit()
            
            QMessageBox.information(
                self,
                "Settings Applied",
                f"TTS adapter has been changed to {adapter_info['name']}.\n\n"
                "Please restart Abogen for the voice list to update and settings to take full effect."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {str(e)}")


class TTSAdapterStatusWidget(QWidget):
    """Widget to display TTS adapter status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.registry = TTSAdapterRegistry.get_instance()
        self.config_manager = get_tts_config_manager()
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Status indicator
        self.status_label = QLabel()
        self.update_status()
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        
        self.setLayout(layout)

    def update_status(self):
        """Update the status display."""
        try:
            tts_config = self.config_manager.get_configuration()
            adapter_id = tts_config.active_adapter
            
            available_adapters = self.registry.get_available_adapters()
            adapter_name = None
            
            for adapter_info in available_adapters:
                if adapter_info['id'] == adapter_id:
                    adapter_name = adapter_info['name']
                    break
            
            if adapter_name:
                voice_count = len(VoiceManager.get_available_voices())
                text = f"TTS: {adapter_name} ({voice_count} voices)"
                self.status_label.setText(text)
                self.status_label.setStyleSheet("color: green;")
            else:
                self.status_label.setText("TTS: Unknown Adapter")
                self.status_label.setStyleSheet("color: orange;")
        except Exception as e:
            self.status_label.setText("TTS: Error")
            self.status_label.setStyleSheet("color: red;")
