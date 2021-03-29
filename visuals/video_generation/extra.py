import pygame , pandas as pd , time , pickle , numpy as np, seaborn as sns , matplotlib.pyplot as plt
from PIL import Image
import io
class Grid:
    
    def __init__(self,xblocks,yblocks,cornerx,cornery):
        self.cornerx = cornerx
        self.cornery = cornery
        self.xblocks = xblocks
        self.yblocks = yblocks
        self.visited = [([0]*xblocks) for i in range(yblocks)]
        for x in range(len(self.visited)):
            for z in range(len(self.visited[0])):
                self.visited[x][z] = (0,0,0)
        self.movenumber = 0
        self.gridnumber = 0
        self.playerloc = []
        self.arrows = []
        self.moves = []
        self.colors = [(0,155,155),(0,255,0),(0,0,255)]
        self.newmap = None
        self.im = None
 
    def addArrows(self,arrows):
        self.arrows = arrows
        
    def addMoves(self,moves):
        self.moves = moves
        
    def addHeat(self,grid):
        self.grid = grid
        
    def addMap(self,newmap):
        self.newmap = newmap
        
    def colorDecay(self):
        for x in range(len(self.visited)):
            for z in range(len(self.visited[0])):
                self.visited[x][z] = (self.visited[x][z][0] * 0.9,self.visited[x][z][1] * 0.9,self.visited[x][z][2] * 0.9)
        
    def draw(self):
        
        #print("gridnumber" + str(self.gridnumber))
        
        if self.newmap is not None and self.gridnumber + 2 > len(self.newmap):
            self.gridnumber = self.gridnumber - 1
        
        if self.newmap is not None:
            self.newmap[self.gridnumber] = np.array(self.newmap[self.gridnumber]).reshape(-1,25)
            if len(self.newmap) is not 1 or self.im is None:
                sns.heatmap(self.newmap[self.gridnumber])
                buf = io.BytesIO()
                plt.savefig(buf,format='png')
                plt.clf()
                buf.seek(0)
                im = Image.open(buf)
                mode = im.mode
                size = im.size
                data = im.tobytes()
                im = pygame.image.fromstring(data, size, mode)
                im = im.subsurface((80,58,397,370))
                im = pygame.transform.scale(im,(WIDTH * xblocks + MARGIN * (xblocks + 1),HEIGHT * yblocks + MARGIN * (yblocks + 1)))
                self.im = im
                buf.close()
            screen.blit(self.im, (self.cornerx,self.cornery))
            
        for row in range(self.yblocks):
            for column in range(self.xblocks):
                if (row,column) in self.playerloc:
                    color = self.colors[self.playerloc.index((row,column))]
                    pygame.draw.rect(screen,
                                     color,
                                     [(MARGIN + WIDTH) * column + MARGIN + self.cornerx,
                                     (MARGIN + HEIGHT) * row + MARGIN + self.cornery,
                                     WIDTH,
                                     HEIGHT])
                elif self.visited[row][column] != (0,0,0):
                    color = self.visited[row][column]
                    pygame.draw.rect(screen,
                                     color,
                                     [(MARGIN + WIDTH) * column + MARGIN + self.cornerx,
                                     (MARGIN + HEIGHT) * row + MARGIN + self.cornery,
                                     WIDTH,
                                     HEIGHT])
        self.colorDecay()
        self.gridnumber += 1
        
    def move(self):
        self.playerloc = []
        for x in range(len(self.moves)):
            self.visited[self.moves[x][self.movenumber][0]][self.moves[x][self.movenumber][1]] = self.colors[x]
            self.playerloc.append((self.moves[x][self.movenumber][0],self.moves[x][self.movenumber][1]))
            self.arrows[x].image = pygame.transform.rotate(self.arrows[x].image,self.moves[x][self.movenumber][2] * 45)
            self.arrows[x].rotation = self.moves[x][self.movenumber][2] * 45
            column = self.moves[x][self.movenumber][1]
            row = self.moves[x][self.movenumber][0]
            self.arrows[x].xy = (MARGIN + WIDTH) * column + MARGIN + WIDTH * 0.5 + self.cornerx,(MARGIN + HEIGHT) * row + MARGIN + HEIGHT * 0.5 + self.cornery
        self.movenumber += 1
        self.draw()

class Player(pygame.sprite.Sprite):
    
    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('arrow1.png')
        self.image = pygame.transform.scale(self.image, (40, 40))
        self.rect = self.image.get_rect()
        self.xy = x , y
        self.rect.center = self.xy
        self.original_image = self.image
        self.rotation = 0
        
    def update(self):
        self.image = pygame.transform.rotate(self.original_image,self.rotation)
        self.rect = self.image.get_rect()
        self.rect.center = self.xy
        
ground_truth = pd.read_csv(r"C:\Users\branc\OneDrive\Desktop\dataFile_ell25_50by50.csv",delim_whitespace=True).values
#moves = pickle.load( open( "newstarts/paths", "rb" ) )
moves = pickle.load( open( "samestarts/paths", "rb" ) )

# Define some colors
BLACK = (100, 100, 100)
WHITE = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 128) 
 
# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 10
HEIGHT = 10

#dimensions
xblocks = 25
yblocks = 25
xgrids = 4
ygrids = 1
 
# This sets the margin between each cell
MARGIN = 1
# This sets the margin between each grid
GRIDMARGIN = 30

WINDOW_SIZE = [(WIDTH * xblocks + MARGIN * (xblocks + 1))* xgrids + GRIDMARGIN * (xgrids + 1)
               ,(HEIGHT * yblocks + MARGIN * (yblocks + 1)) * ygrids + GRIDMARGIN * (ygrids + 1)]

# This will be the spacing between grid topleft corner
XGRIDSPACING = WIDTH * xblocks + MARGIN * (xblocks + 1) + GRIDMARGIN
YGRIDSPACING = HEIGHT * yblocks + MARGIN * (yblocks + 1) + GRIDMARGIN

# Initialize pygame
pygame.init()

grids = []
all_sprites = pygame.sprite.Group()

for row in range(xgrids):
    # Add an empty array that will hold each cell
    # in this row
    grids.append([])
    for column in range(ygrids):
        x = GRIDMARGIN + (XGRIDSPACING * row)
        y = GRIDMARGIN + (YGRIDSPACING * column)
        grids[row].append(Grid(xblocks,yblocks,x,y))


"""
lists of heatmaps will be inputted below
"""

ground_truth = ground_truth[0:25,0:25]
ground_truth = ground_truth.flatten()
heat = []
heat.append(ground_truth)
"""
ax = sns.heatmap(ground_truth)
#gtf = ax.get_figure()
#plt.show()
buf = io.BytesIO()
plt.savefig(buf,format='png')
buf.seek(0)
im = Image.open(buf)
im.show()
buf.close()
"""

#ests = pickle.load( open( "newstarts/ests", "rb" ) )

ests = pickle.load( open( "samestarts/ests", "rb" ) )

#stds = pickle.load( open( "stds", "rb" ) )
temp = []
temp.append(ground_truth)

for x in range(len(moves)):
    all_sprites.add(Player(GRIDMARGIN * 0.5 + XGRIDSPACING * row,GRIDMARGIN * 0.5 + YGRIDSPACING * column))
grids[0][0].addArrows(all_sprites.sprites())
grids[0][0].addMoves(moves)
grids[0][0].addMap(temp)
grids[1][0].addMap(ests[0])
grids[2][0].addMap(ests[1])
grids[3][0].addMap(ests[2])

# Set the HEIGHT and WIDTH of the screen
screen = pygame.display.set_mode(WINDOW_SIZE)

#captions
font = pygame.font.Font('freesansbold.ttf',20)
text = []
text.append(font.render('Agents',True,GREEN,BLUE))

text.append(font.render('Temperature estimate',True,(0,155,155),BLACK))
text.append(font.render('Temperature estimate',True,(0,255,0),BLACK))
text.append(font.render('Temperature estimate',True,(0,0,255),BLACK))

#[(0,155,155),(0,255,0),(0,0,255)]

textRect = []

print(len(grids))

for x in range(len(grids)):
    textRect.append(text[x].get_rect())
    
# Set title of screen
pygame.display.set_caption("Mean field games")
 
# Loop until the user clicks the close button.
done = False
 
# Used to manage how fast the screen updates
clock = pygame.time.Clock()
movecount = 0

while not done:
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
        elif event.type == pygame.MOUSEBUTTONDOWN:
            done = True
            
    clock.tick(30)
    
done = False

# -------- Main Program Loop -----------
while not done:
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
            
    # Set the screen background
    screen.fill(BLACK)
    
    if movecount < len(moves[0]):
        for x in grids:
            for y in x:
                y.move()
                
    if movecount < len(moves[0]):
        for x in range(len(grids)):
            textRect[x].bottomleft = (grids[x][0].cornerx,grids[x][0].cornery-5)
            screen.blit(text[x], textRect[x]) 
        all_sprites.update()
        all_sprites.draw(screen)
        pygame.display.flip()
        
    if movecount == 4:
        ground_truth = 0
        
    #print("move number:" + str(movecount))
    movecount += 1
    # Limit to 60 frames per second
    clock.tick(60)
    #time.sleep(.2)
 
    # Go ahead and update the screen with what we've drawn.
    #pygame.display.flip()
 
pygame.quit()