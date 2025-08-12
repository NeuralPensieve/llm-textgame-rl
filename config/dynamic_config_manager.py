# dynamic_config_manager.py
import json
import os
import time
from pathlib import Path
from typing import Dict, Any
import logging

class DynamicConfigManager:
    def __init__(self, static_config, logger=None):
        self.static_config = static_config
        self.logger = logger or logging.getLogger(__name__)
        self.config_file = Path("config/dynamic_params.json")
        self.last_modified = 0
        
        # Initialize dynamic parameters from static config
        self.dynamic_params = {
            'value_loss_coef': static_config.value_loss_coef,
            'entropy_coef': static_config.entropy_coef,
        }
        
        # Create initial dynamic params file if it doesn't exist
        if not self.config_file.exists():
            self.save_current_params("Initial dynamic parameters")
    
    def save_current_params(self, note=""):
        """Save current dynamic params to file"""
        output = {
            "parameters": self.dynamic_params,
            "note": note,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.config_file, 'w') as f:
            json.dump(output, f, indent=2)
    
    def check_and_update(self) -> Dict[str, Any]:
        """Check for updates and return dict of changes"""
        changes = {}
        
        try:
            # Check if file has been modified
            stat = os.stat(self.config_file)
            if stat.st_mtime <= self.last_modified:
                return changes
            
            # Load new parameters
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                new_params = data.get("parameters", {})
            
            # Identify what changed
            for key, new_value in new_params.items():
                if key in self.dynamic_params:
                    old_value = self.dynamic_params[key]
                    if old_value != new_value:
                        changes[key] = (old_value, new_value)
                        self.dynamic_params[key] = new_value
                        if self.logger:
                            self.logger.info(f"Updated {key}: {old_value} -> {new_value}")
            
            self.last_modified = stat.st_mtime
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error reading dynamic config: {e}")
        
        return changes
    
    def get(self, key: str, default=None):
        """Get a dynamic parameter value"""
        return self.dynamic_params.get(key, default)