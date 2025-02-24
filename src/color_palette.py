import colorsys
import json
import os
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class ColorPaletteManager:
    def __init__(self):
        self.default_palette = {
            'joy': '#FFD700',        # Gold
            'love': '#FF69B4',       # Hot Pink
            'excitement': '#FF4500',  # Orange Red
            'optimism': '#98FB98',   # Pale Green
            'pride': '#9370DB',      # Medium Purple
            'neutral': '#808080',    # Gray
            'sadness': '#4682B4',    # Steel Blue
            'anger': '#DC143C',      # Crimson
            'fear': '#800080',       # Purple
            'grief': '#2F4F4F'       # Dark Slate Gray
        }
        self.current_palette = self.default_palette.copy()

    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color."""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    def get_intensity_variant(self, base_color: str, intensity: float) -> str:
        """Generate color variant based on emotional intensity."""
        try:
            # Convert hex to RGB
            rgb = self.hex_to_rgb(base_color)
            
            # Convert RGB to HSV
            hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            
            # Adjust saturation and value based on intensity
            # Keep hue constant, adjust saturation and value
            new_hsv = (
                hsv[0],  # Hue stays the same
                min(1.0, hsv[1] * (0.5 + intensity/2)),  # Saturation
                min(1.0, hsv[2] * (0.5 + intensity/2))   # Value
            )
            
            # Convert back to RGB
            rgb = colorsys.hsv_to_rgb(*new_hsv)
            
            # Convert to hex
            return '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
        except Exception as e:
            logger.error(f"Error generating color variant: {str(e)}")
            return base_color

    def update_emotion_color(self, emotion: str, color: str) -> None:
        """Update color for a specific emotion."""
        if emotion in self.current_palette:
            self.current_palette[emotion] = color

    def get_emotion_color(self, emotion: str, intensity: float = 1.0) -> str:
        """Get color for emotion with intensity adjustment."""
        base_color = self.current_palette.get(emotion, self.default_palette.get(emotion, '#808080'))
        if intensity == 1.0:
            return base_color
        return self.get_intensity_variant(base_color, intensity)

    def get_complementary_colors(self, emotion: str) -> List[str]:
        """Generate complementary colors for an emotion."""
        try:
            base_rgb = self.hex_to_rgb(self.get_emotion_color(emotion))
            base_hsv = colorsys.rgb_to_hsv(base_rgb[0]/255, base_rgb[1]/255, base_rgb[2]/255)
            
            complementary_colors = []
            
            # Generate 3 complementary colors with different hue shifts
            for hue_shift in [0.33, 0.66, 0.5]:  # 120°, 240°, 180° shifts
                new_hue = (base_hsv[0] + hue_shift) % 1.0
                new_hsv = (new_hue, base_hsv[1], base_hsv[2])
                new_rgb = colorsys.hsv_to_rgb(*new_hsv)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(new_rgb[0] * 255),
                    int(new_rgb[1] * 255),
                    int(new_rgb[2] * 255)
                )
                complementary_colors.append(hex_color)
            
            return complementary_colors
        except Exception as e:
            logger.error(f"Error generating complementary colors: {str(e)}")
            return ['#808080', '#A9A9A9', '#C0C0C0']  # Fallback to grayscale

    def reset_to_default(self) -> None:
        """Reset palette to default colors."""
        self.current_palette = self.default_palette.copy()

    def get_color_scheme(self, emotion: str) -> Dict[str, str]:
        """Get a complete color scheme for an emotion."""
        base_color = self.get_emotion_color(emotion)
        complementary = self.get_complementary_colors(emotion)
        
        return {
            'primary': base_color,
            'complementary': complementary,
            'light': self.get_intensity_variant(base_color, 0.5),
            'dark': self.get_intensity_variant(base_color, 1.5)
        }
