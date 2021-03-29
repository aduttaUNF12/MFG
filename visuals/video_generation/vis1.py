'''
UNF
@author: JT Wolbert

DOC:
Visualization(moves,heatmaps,exportDirectory,suppressOutput,suppressVideo,agentsTraced,dataDisplayed,dimensions):
param#1:moves, a list of lists that is the agents' (x,y) 2-tuples for each step of the episode
param#2:heatmaps, a list of lists of 2d-arrays starting with ground truth and followed by the agents' perspectives
(the ground truth must be in a list by itself at index 0 of heatmaps)
param#3:exportDirectory, a video will be output to this directory after each frame is rendered
param#4:suppressOuput, assert to prevent popup window
param#5:suppressVideo, assert to skip creation of the video
(its best to leave this deasserted for right now because the video is the only useful output currently)
param#6:agentsTraced, the module supports two grids that depict two agents specified by the user in a tuple
(agents are indexed here starting at 1!)
param#7:dataDisplayed, this is simply the caption for the traced agent grids
(most likely going to be "Entopy" or "Temperature estimate")
param#8:dimensions, this will be the x,y dimensions of the grid world given as a 2-tuple

In an effort to not overwhelm the viewer with information the number of grids was reduced to 3.
The first grid shows agent locations overlaid on top of the ground-truth data.
The second and third grids will show perspectives of 2 selected agents.
(The visualization function still takes ALL the heatmaps for each agent so,
if there's 30 agents there will be 31(agents + groundtruth map) heatmaps going into the function.
The additional unseen maps are not rendered so the only wasted resources should be memory,
and this is something that I can change later.)

The mean field arrows were also subtracted temporarily for the same reason.

'''

import pygame , pandas as pd , pickle , numpy as np, matplotlib.pyplot as plt
import random
import imageio , os

class Visualization(object):


    def __init__(self,moves,heatmaps,exportDirectory = "",suppressOutput = 1,suppressVideo = 1,agentsTraced = (1,2),datadisplayed = "Temperature Estimate",dimensions = (25,25)):
        self.moves = moves
        self.heatmaps = heatmaps
        if exportDirectory is not "":
            exportDirectory = exportDirectory + "/"
        self.exportDirectory = exportDirectory
        self.suppressOutput = suppressOutput
        self.suppressVideo = suppressVideo
        self.agentsTraced = agentsTraced
        self.datadisplayed = datadisplayed
        self.dimensions = dimensions 
        
    def visualize(self):
        
        print("Starting visualization...")
        
        # Define some colors
        GREY = (100, 100, 100)
        GREEN = ( 50, 255, 50)
        BLUE = (25, 25, 150) 
         
        # This sets the WIDTH and HEIGHT of each grid location and player
        WIDTH = 14
        HEIGHT = 14
        #history indent, previous location rectangle size
        hsize = 6
        
        #dimensions of grid
        xblocks = self.dimensions[0]
        yblocks = self.dimensions[1]
        xgrids = 3
        """
        if len(self.heatmaps) % xgrids is 0:
            ygrids = len(self.heatmaps) // xgrids
        else:
            ygrids = len(self.heatmaps) // xgrids + 1
        """
        ygrids = 1
        # This sets the margin between each cell
        MARGIN = 1
        # This sets the margin between each grid
        GRIDMARGIN = 30
        #correction for imageio
        correctspac = 2
        
        WINDOW_SIZE = [(WIDTH * xblocks + MARGIN * (xblocks + 1))* xgrids + GRIDMARGIN * (xgrids + 1) + correctspac
                       ,(HEIGHT * yblocks + MARGIN * (yblocks + 1)) * ygrids + GRIDMARGIN * (ygrids + 1) + correctspac]
        
        # This will be the spacing between grid topleft corner
        XGRIDSPACING = WIDTH * xblocks + MARGIN * (xblocks + 1) + GRIDMARGIN
        YGRIDSPACING = HEIGHT * yblocks + MARGIN * (yblocks + 1) + GRIDMARGIN
        
        #correction for imageio
        WINDOW_SIZE  = (WINDOW_SIZE[0] + (16 - WINDOW_SIZE[0] % 16),WINDOW_SIZE[1] + (16 - WINDOW_SIZE[1] % 16))
        
        # Initialize pygame
        pygame.init()
        
        if self.suppressOutput is 1:
            screen = pygame.Surface(WINDOW_SIZE)
        else:
            screen = pygame.display.set_mode(WINDOW_SIZE)
        
        grids = []
        #all_sprites = pygame.sprite.Group()
        
        for row in range(xgrids):
            # Add an empty array that will hold each cell
            # in this row
            grids.append([])
            for column in range(ygrids):
                x = GRIDMARGIN + (XGRIDSPACING * row)
                y = GRIDMARGIN + (YGRIDSPACING * column)
                grids[row].append(Grid(xblocks,yblocks,x,y,screen,(WIDTH,HEIGHT,MARGIN,hsize),self.agentsTraced)) 
        """
        for x in range(len(self.moves)):
            all_sprites.add(Player(GRIDMARGIN * 0.5 + XGRIDSPACING * row,GRIDMARGIN * 0.5 + YGRIDSPACING * column))
        """
        
        #grids[0][0].addArrows(all_sprites.sprites())
        grids[0][0].addMoves(self.moves)
        
        #updated
        grids[0][0].addMap(self.heatmaps[0])
        grids[1][0].addMap(self.heatmaps[self.agentsTraced[0]])
        grids[2][0].addMap(self.heatmaps[self.agentsTraced[1]])
        
        #captions
        font = pygame.font.Font('freesansbold.ttf',20)
        text = []
        text.append(font.render('Agents',True,GREEN,BLUE))

        #updated
        text.append(font.render(self.datadisplayed + " " + str(1),True,grids[0][0].colors[self.agentsTraced[0] - 1],GREY))
        text.append(font.render(self.datadisplayed + " " + str(2),True,grids[0][0].colors[self.agentsTraced[1] - 1],GREY))
        
        textRect = []
        
        for x in range(xgrids):
            textRect.append(text[x].get_rect())
            
        # Set title of screen
        if self.suppressOutput is 0:
            pygame.display.set_caption("Multi-robot Informative Sampling")
         
        # Loop until the user clicks the close button.
        done = False
         
        # Used to manage how fast the screen updates
        movecount = 0
        
        while not done:
            if movecount != 0 and movecount % 10 == 0:
                print("Starting move " + str(movecount) + " of " + str(len(moves[0])) + " ...")
            if self.suppressOutput is 0:
                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        done = True  # Flag that we are done so we exit this loop
    
            if movecount < len(self.moves[0]):
                screen.fill(GREY)
                for x in grids:
                    for y in x:
                        y.move()
                        #all_sprites.update()
                        if self.suppressVideo is 0:
                            y.draw()
                            #all_sprites.draw(screen)
                            
            if movecount == len(self.moves[0]) - 1:
                screen.fill(GREY)
                self.suppressVideo = 0
                for x in grids:
                    for y in x:
                        if y.newmap is not None:
                            y.gridnumber = len(y.newmap) - 1
                            y.draw()
                            #all_sprites.draw(screen)
                    
            if movecount < len(self.moves[0]) and self.suppressVideo is 0:
                for x in range(len(textRect)):
                    textRect[x].bottomleft = (grids[x % xgrids][x // xgrids].cornerx,grids[x % xgrids][x // xgrids].cornery-5)
                    screen.blit(text[x], textRect[x])
                pygame.image.save(screen,self.exportDirectory + "frame" + str(movecount) + ".png")
              
            if self.suppressOutput is 0:      
                pygame.display.flip()
            
            if movecount == len(self.moves[0]) - 1:
                #pygame.image.save(screen,self.exportDirectory + "out.png")
                pygame.quit()
                done = True
            
            movecount += 1
        
        print("Saving output.")
        if self.suppressVideo == 1:
            filenames = []
            for x in range(len(self.moves[0])):
                filenames.append(self.exportDirectory + "frame" + str(x) + ".png")
            images = []
            for filename in filenames:
                images.append(imageio.imread(filename))
                
            imageio.mimsave(self.exportDirectory + "anim.mp4", images)
            for filename in filenames:
                os.remove(filename)
 
        pygame.quit()
        

class Grid:
    
    def __init__(self,xblocks,yblocks,cornerx,cornery,screen,dims,agentsTraced):
        self.cornerx = cornerx
        self.cornery = cornery
        self.xblocks = xblocks
        self.yblocks = yblocks
        self.visited = [([0]*xblocks) for i in range(yblocks)]
        for x in range(len(self.visited)):
            for z in range(len(self.visited[0])):
                self.visited[x][z] = (0,0,0,0)
        self.movenumber = 0
        self.gridnumber = 0
        self.playerloc = []
        self.arrows = []
        self.moves = []
        self.colors = []
        self.newmap = None
        self.im = None
        self.screen = screen
        self.WIDTH = dims[0]
        self.HEIGHT = dims[1]
        self.MARGIN = dims[2]
        self.hsize = dims[3]
        self.newmap = None
        self.font = pygame.font.Font('freesansbold.ttf',self.WIDTH)
        self.agentsTraced = agentsTraced
 
    def addArrows(self,arrows):
        self.arrows = arrows
        
    def addMoves(self,moves):
        self.moves = moves
        for x in moves:
            self.colors.append((random.random() * 155 + 100,random.random() * 155 + 100,random.random() * 155 + 100))
        random.random()
        
    def addHeat(self,grid):
        self.grid = grid
        
    def addMap(self,newmap):
        self.newmap = newmap
        
    def colorDecay(self):
        for x in range(len(self.visited)):
            for z in range(len(self.visited[0])):
                if self.visited[x][z][3] < 100:
                    self.visited[x][z] = (0,0,0,0)
                else:
                    self.visited[x][z] = (self.visited[x][z][0],self.visited[x][z][1],self.visited[x][z][2],self.visited[x][z][3] * .9)
        
    def draw(self):
             
        if self.newmap is not None and self.gridnumber + 2 > len(self.newmap):
            self.gridnumber = self.gridnumber - 1
        
        if self.newmap is not None:
            self.newmap[self.gridnumber] = np.array(self.newmap[self.gridnumber]).reshape(-1,25)
            if len(self.newmap) is not 1 or self.im is None:
                im = plt.imshow(self.newmap[self.gridnumber])
                color_matrix = im.cmap(im.norm(im.get_array()))
                plt.clf()
            
        for row in range(self.yblocks):
            for column in range(self.xblocks):
                color = (color_matrix[column][row][0] * 255,color_matrix[column][row][1] * 255,color_matrix[column][row][2] * 255)
                pygame.draw.rect(self.screen,
                                     color,
                                     [(self.MARGIN + self.WIDTH) * column + self.MARGIN + self.cornerx,
                                     (self.MARGIN + self.HEIGHT) * row + self.MARGIN + self.cornery,
                                     self.WIDTH,
                                     self.HEIGHT])

                if (row,column) in self.playerloc:
                    color = self.colors[self.playerloc.index((row,column))]
                    xdest = (self.MARGIN + self.WIDTH) * column + self.MARGIN + self.cornerx
                    ydest = (self.MARGIN + self.HEIGHT) * row + self.MARGIN + self.cornery
                    pygame.draw.rect(self.screen,
                                     color,
                                     [xdest,
                                     ydest,
                                     self.WIDTH,
                                     self.HEIGHT])
                    if self.playerloc.index((row,column)) == self.agentsTraced[0] - 1 or self.playerloc.index((row,column)) == self.agentsTraced[1] - 1:
                        newtext = self.font.render(str(self.playerloc.index((row,column))),True,(0,0,0))
                        #add width/4 to help algn text
                        self.screen.blit(newtext,(xdest + self.WIDTH//4,ydest))
                elif self.visited[row][column] != (0,0,0,100):
                    color = (self.visited[row][column][0],self.visited[row][column][1],self.visited[row][column][2])
                    s = pygame.Surface((self.hsize,self.hsize))  # the size of your rect
                    s.set_alpha(self.visited[row][column][3])                # alpha level
                    s.fill(color)           # this fills the entire surface
                    xdest = (self.MARGIN + self.WIDTH) * column + self.MARGIN + self.cornerx + (self.WIDTH - self.hsize)/2
                    ydest = (self.MARGIN + self.HEIGHT) * row + self.MARGIN + self.cornery + (self.WIDTH - self.hsize)/2
                    self.screen.blit(s, (xdest,ydest) )
            
        self.colorDecay()
        self.gridnumber += 1
        
    def move(self):
        self.playerloc = []
        for x in range(len(self.moves)):
            l = list(self.colors[x])
            l.append(255)
            t = tuple(l)
            self.visited[self.moves[x][self.movenumber][0]][self.moves[x][self.movenumber][1]] = t
            self.playerloc.append((self.moves[x][self.movenumber][0],self.moves[x][self.movenumber][1]))
            #below code is for mean field arrows
            #self.arrows[x].image = pygame.transform.rotate(self.arrows[x].image,self.moves[x][self.movenumber][2] * 45)
            #self.arrows[x].rotation = self.moves[x][self.movenumber][2] * 45
            column = self.moves[x][self.movenumber][1]
            row = self.moves[x][self.movenumber][0]
            #self.arrows[x].xy = (self.MARGIN + self.WIDTH) * column + self.MARGIN + self.WIDTH * 0.5 + self.cornerx,(self.MARGIN + self.HEIGHT) * row + self.MARGIN + self.HEIGHT * 0.5 + self.cornery
        self.movenumber += 1

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
        
        
        
#test bench
ground_truth = pd.read_csv(r"C:\Users\branc\OneDrive\Desktop\dataFile_ell25_50by50.csv",delim_whitespace=True).values
moves = pickle.load( open( "samestarts/paths", "rb" ) )
ground_truth = ground_truth[0:25,0:25]
ground_truth = ground_truth.flatten()
ests = pickle.load( open( "samestarts/ests", "rb" ) )
temp = []
temp.append(ground_truth)
temp1 = []
temp1.append(temp)
for x in ests:
    temp1.append(x)

new = []
for x in moves:
    temp = []
    for y in x:
        temp.append((y[0],y[1]))
    new.append(temp)
    
v = Visualization(new,temp1,r"C:\Users\branc\eclipse-workspace\newgrid\mygrid\video",1,1,(2,3),"Temperature Estimate")
v.visualize()
