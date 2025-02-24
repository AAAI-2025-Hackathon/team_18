import streamlit as st
from typing import Dict, Any, Optional
import colorsys
from dataclasses import dataclass
from src.color_palette import ColorPaletteManager

@dataclass
class UIPreferences:
    theme_color: str
    font_size: str
    animation_speed: str
    layout_density: str
    interaction_style: str

class UIPersonalizer:
    def __init__(self):
        self.color_manager = ColorPaletteManager()
        self.default_preferences = UIPreferences(
            theme_color="#FF4B4B",
            font_size="medium",
            animation_speed="normal",
            layout_density="comfortable",
            interaction_style="standard"
        )
    
    def generate_theme(self, emotion: str, intensity: float) -> Dict[str, str]:
        """Generate a theme based on emotional state."""
        color_scheme = self.color_manager.get_color_scheme(emotion)
        
        # Adjust saturation and brightness based on intensity
        base_color = color_scheme['primary']
        rgb = self._hex_to_rgb(base_color)
        hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        
        # Adjust saturation and value based on intensity
        adjusted_hsv = (
            hsv[0],
            min(1.0, hsv[1] * (0.5 + intensity/2)),
            min(1.0, hsv[2] * (0.5 + intensity/2))
        )
        
        adjusted_rgb = colorsys.hsv_to_rgb(*adjusted_hsv)
        primary_color = '#{:02x}{:02x}{:02x}'.format(
            int(adjusted_rgb[0] * 255),
            int(adjusted_rgb[1] * 255),
            int(adjusted_rgb[2] * 255)
        )
        
        return {
            'primaryColor': primary_color,
            'backgroundColor': color_scheme['light'],
            'secondaryBackgroundColor': color_scheme['complementary'][0],
            'textColor': self._get_text_color(primary_color)
        }
    
    def adjust_layout(self, emotion: str, intensity: float) -> Dict[str, Any]:
        """Adjust layout based on emotional state."""
        # Map emotions to layout preferences
        layout_map = {
            'joy': {'spacing': 'loose', 'width': 'wide'},
            'sadness': {'spacing': 'tight', 'width': 'narrow'},
            'anger': {'spacing': 'loose', 'width': 'full'},
            'fear': {'spacing': 'comfortable', 'width': 'narrow'},
            'neutral': {'spacing': 'comfortable', 'width': 'medium'}
        }
        
        base_layout = layout_map.get(emotion, layout_map['neutral'])
        
        # Adjust based on intensity
        padding = max(1.0, min(2.0, 1.0 + intensity * 0.5))
        
        return {
            'spacing': base_layout['spacing'],
            'width': base_layout['width'],
            'padding': f"{padding}rem",
            'animation_duration': self._get_animation_duration(emotion, intensity)
        }
    
    def get_interaction_style(self, emotion: str, intensity: float) -> Dict[str, Any]:
        """Define interaction style based on emotional state."""
        base_styles = {
            'joy': {
                'button_style': 'rounded',
                'hover_effect': 'bounce',
                'transition_speed': 'fast'
            },
            'sadness': {
                'button_style': 'soft',
                'hover_effect': 'gentle',
                'transition_speed': 'slow'
            },
            'anger': {
                'button_style': 'sharp',
                'hover_effect': 'shake',
                'transition_speed': 'fast'
            },
            'fear': {
                'button_style': 'subtle',
                'hover_effect': 'fade',
                'transition_speed': 'slow'
            },
            'neutral': {
                'button_style': 'standard',
                'hover_effect': 'standard',
                'transition_speed': 'normal'
            }
        }
        
        style = base_styles.get(emotion, base_styles['neutral'])
        
        # Adjust transition speed based on intensity
        speed_map = {
            'slow': 0.5,
            'normal': 0.3,
            'fast': 0.15
        }
        base_speed = speed_map[style['transition_speed']]
        adjusted_speed = base_speed * (1 + (1 - intensity) * 0.5)
        
        return {
            **style,
            'transition_duration': f"{adjusted_speed}s"
        }
    
    def apply_preferences(self, emotion: str, intensity: float):
        """Apply all UI preferences based on emotional state."""
        theme = self.generate_theme(emotion, intensity)
        layout = self.adjust_layout(emotion, intensity)
        interaction = self.get_interaction_style(emotion, intensity)
        
        # Apply theme to Streamlit
        st.markdown(f"""
            <style>
                /* Theme Colors */
                :root {{
                    --primary-color: {theme['primaryColor']};
                    --background-color: {theme['backgroundColor']};
                    --secondary-bg-color: {theme['secondaryBackgroundColor']};
                    --text-color: {theme['textColor']};
                }}
                
                /* Layout Adjustments */
                .stApp {{
                    padding: {layout['padding']};
                }}
                
                /* Interactive Elements */
                .stButton > button {{
                    border-radius: {self._get_button_radius(interaction['button_style'])};
                    transition: all {interaction['transition_duration']} ease-in-out;
                }}
                
                .stButton > button:hover {{
                    transform: {self._get_hover_transform(interaction['hover_effect'])};
                }}
            </style>
        """, unsafe_allow_html=True)
    
    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _get_text_color(self, background_color: str) -> str:
        """Determine appropriate text color based on background."""
        rgb = self._hex_to_rgb(background_color)
        brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
        return '#000000' if brightness > 128 else '#FFFFFF'
    
    def _get_animation_duration(self, emotion: str, intensity: float) -> str:
        """Calculate animation duration based on emotional state."""
        base_duration = 300  # milliseconds
        emotion_multiplier = {
            'joy': 0.8,
            'sadness': 1.5,
            'anger': 0.6,
            'fear': 1.2,
            'neutral': 1.0
        }.get(emotion, 1.0)
        
        # Adjust duration based on intensity
        adjusted_duration = base_duration * emotion_multiplier * (1 + (1 - intensity) * 0.5)
        return f"{int(adjusted_duration)}ms"
    
    def _get_button_radius(self, style: str) -> str:
        """Get button border radius based on style."""
        radius_map = {
            'rounded': '20px',
            'soft': '10px',
            'sharp': '2px',
            'subtle': '5px',
            'standard': '4px'
        }
        return radius_map.get(style, '4px')
    
    def _get_hover_transform(self, effect: str) -> str:
        """Get hover transform effect."""
        transform_map = {
            'bounce': 'scale(1.05)',
            'gentle': 'scale(1.02)',
            'shake': 'translateX(2px)',
            'fade': 'opacity(0.8)',
            'standard': 'scale(1.03)'
        }
        return transform_map.get(effect, 'scale(1.03)')
