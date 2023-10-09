import numpy as np
import pygame

class Screen:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        pygame.init()
        self.screen = pygame.display.set_mode([width, height])

    def ratio(self):
        return self.width / self.height

    def draw(self, buf: np.ndarray):
        """Takes a buffer of 8-bit RGB pixels and puts them on the canvas.
        buf should be a ndarray of shape (height, width, 3)"""
        # Make sure that the buffer is HxWx3
        if buf.shape != (self.height, self.width, 3):
            raise Exception("buffer and screen not the same size")

        # Flip buffer
        buf = np.fliplr(buf)

        # The prefered way to accomplish this
        pygame.pixelcopy.array_to_surface(self.screen, buf)

        # An alternative (slower) way, but still valid
        # Iterate over the pixels and paint them
        # for x, row in enumerate(buf):
            # for y, pix in enumerate(row):
                # self.screen.set_at((x, y), pix.tolist())

        # Update the display
        pygame.display.flip()

    def show(self):
        """Shows the canvas"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()




