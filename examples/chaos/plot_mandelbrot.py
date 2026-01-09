import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.sciforge.chaos.fractals import generate_mandelbrot_tile

class InteractivePlot:
    """A flexible base class for creating interactive plots with pan and zoom."""
    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range
        self.press_pos = None
        self.rect = None

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        # Enforce a 1:1 aspect ratio to prevent distortion
        self.ax.set_aspect('equal', adjustable='box')
        self.connect_events()

    def connect_events(self):
        """Connect matplotlib event handlers."""
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_scroll(self, event):
        """Handle mouse scroll event for zooming."""
        if event.xdata is None or event.ydata is None: return
        zoom_factor = 1.3 if event.button == 'up' else 1 / 1.3
        
        new_width = (self.x_range[1] - self.x_range[0]) / zoom_factor
        new_height = (self.y_range[1] - self.y_range[0]) / zoom_factor
        
        rel_x = (event.xdata - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        rel_y = (event.ydata - self.y_range[0]) / (self.y_range[1] - self.y_range[0])

        self.x_range = (event.xdata - rel_x * new_width, event.xdata + (1 - rel_x) * new_width)
        self.y_range = (event.ydata - rel_y * new_height, event.ydata + (1 - rel_y) * new_height)
        self.update_plot()

    def on_press(self, event):
        """Handle mouse button press for box zoom."""
        if event.inaxes != self.ax or event.button != 1: return
        self.press_pos = (event.xdata, event.ydata)
        self.rect = plt.Rectangle(self.press_pos, 0, 0, facecolor='none', edgecolor='white', lw=1, animated=True)
        self.ax.add_patch(self.rect)

    def on_motion(self, event):
        """Handle mouse motion for drawing the zoom box."""
        if self.press_pos is None or event.inaxes != self.ax: return
        x0, y0 = self.press_pos
        self.rect.set_width(event.xdata - x0)
        self.rect.set_height(event.ydata - y0)
        self.ax.draw_artist(self.rect)
        self.fig.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        """Handle mouse button release to perform the box zoom."""
        if self.press_pos is None or self.rect is None or event.button != 1: return
        
        x0, y0 = self.press_pos
        x1, y1 = event.xdata, event.ydata
        
        self.rect.remove()
        self.rect = None
        self.press_pos = None

        if x1 is None or abs(x0 - x1) < 1e-6:
             self.fig.canvas.draw()
             return

        self.x_range = tuple(sorted((x0, x1)))
        self.y_range = tuple(sorted((y0, y1)))
        self.update_plot()
        
    def update_plot(self):
        """This method should be implemented by subclasses to redraw the plot."""
        raise NotImplementedError

class MandelbrotViewer(InteractivePlot):
    """An optimized, interactive viewer for the Mandelbrot set using tiling and caching."""
    def __init__(self, tile_size=256):
        super().__init__(x_range=(-2.0, 1.0), y_range=(-1.5, 1.5))
        self.tile_size = tile_size
        self.tile_cache = {}
        self.ax.set_facecolor('black')
        self.ax.set_title("Mandelbrot Set Explorer")
        self.update_plot()
        plt.show()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(self.x_range)
        self.ax.set_ylim(self.y_range)
        self.ax.set_xlabel("Re(c)")
        self.ax.set_ylabel("Im(c)")
        
        view_width_complex = self.x_range[1] - self.x_range[0]
        view_height_complex = self.y_range[1] - self.y_range[0]
        max_iter = int(max(128, 50 * abs(np.log10(view_width_complex))))

        ax_bbox = self.ax.get_window_extent()
        ax_width_pixels = ax_bbox.width
        ax_height_pixels = ax_bbox.height

        pixel_width_complex = view_width_complex / ax_width_pixels
        pixel_height_complex = view_height_complex / ax_height_pixels

        tile_w_complex = self.tile_size * pixel_width_complex
        tile_h_complex = self.tile_size * pixel_height_complex

        start_col = int(self.x_range[0] / tile_w_complex)
        end_col = int(self.x_range[1] / tile_w_complex)
        start_row = int(self.y_range[0] / tile_h_complex)
        end_row = int(self.y_range[1] / tile_h_complex)

        print(f"Rendering with max_iter={max_iter}. View range: {abs(end_col-start_col)}x{abs(end_row-start_row)} tiles.")
        
        for r in range(start_row, end_row + 1):
            for c in range(start_col, end_col + 1):
                tile_id = (r, c, max_iter)
                if tile_id not in self.tile_cache:
                    tile_x_min = c * tile_w_complex
                    tile_y_min = r * tile_h_complex
                    self.tile_cache[tile_id] = generate_mandelbrot_tile(
                        tile_x_min, tile_y_min, 
                        pixel_width_complex, pixel_height_complex,
                        self.tile_size, self.tile_size, max_iter
                    )
                
                tile_data = self.tile_cache[tile_id]
                self.ax.imshow(np.log(tile_data.T + 1e-9),
                               extent=[c * tile_w_complex, (c + 1) * tile_w_complex,
                                       r * tile_h_complex, (r + 1) * tile_h_complex],
                               origin='lower', cmap='magma', interpolation='nearest')
        self.fig.canvas.draw()

if __name__ == "__main__":
    viewer = MandelbrotViewer() 