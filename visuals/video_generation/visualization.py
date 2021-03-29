'''
UNF
@author: JT Wolbert

DOC:
Visualization(moves,heatmaps,exportDirectory,suppressOutput,suppressVideo,agentsTraced,dataDisplayed,dimensions):
param#1:moves, a list of lists that is the agents' (x,y) 2-tuples for each step of the episode
param#2:heatmaps, a list of lists of lists of 2d-arrays starting with ground truth and followed by the agents' perspectives
(the ground truth must be in a list by itself at index 0 of heatmaps)
param#3:exportDirectory, a video will be output to this directory after each frame is rendered
param#4:suppressOuput, assert to prevent popup windows
param#5:suppressVideo, assert to skip creation of the video but still get coverage plot
param#6:dataDisplayed, these are the captions for the agent heatmap grids, a n-tuple of strings
param#7:dimensions, this will be the x,y dimensions of the grid world given as a 2-tuple
param#8:scale, a single integer which controls the scale of the entire visualization

The first grid shows agent locations overlaid on top of the ground-truth data.
The heatmaps parameter is structured as follows: heatmaps[agent][datatype][movenumber][x][y]
where the dataDisplayed parameter correlates with 'datatype'

For a visual with 15 heatmaps, it will take about 2.5 minutes to render. 

'''

import pygame , pandas as pd , pickle , numpy as np, matplotlib.pyplot as plt
import random
import imageio , os

class Visualization(object):


    def __init__(self,moves,groundTruth,heatmaps,exportDirectory,suppressOutput,suppressVideo,datadisplayed,dimensions,scale):
        self.moves = moves
        self.heatmaps = heatmaps
        if exportDirectory is not "" or exportDirectory[-1] is not "/":
            exportDirectory = exportDirectory + "/"
        self.exportDirectory = exportDirectory
        self.suppressOutput = suppressOutput
        self.suppressVideo = suppressVideo
        self.datadisplayed = datadisplayed
        self.dimensions = dimensions
        self.groundTruth = []
        self.groundTruth.append(groundTruth)
        self.scale = scale
        
    def visualize(self):
        
        if self.suppressVideo == 1:
            self.coverage()
            return
        
        print("Getting visualization...", end='')
        
        # Define some colors
        GREY = (100, 100, 100)
        GREEN = ( 50, 255, 50)
        BLUE = (25, 25, 150) 
         
        # This sets the WIDTH and HEIGHT of each grid location and player
        WIDTH = self.scale
        HEIGHT = self.scale
        #history indent, previous location rectangle size
        hsize = self.scale
        
        #dimensions of grid
        xblocks = self.dimensions[0]
        yblocks = self.dimensions[1]
        ygrids = len(self.heatmaps)
        xgrids = len(self.heatmaps[0])
        # This sets the margin between each cell
        MARGIN = self.scale // 11 + 1
        # This sets the margin between each grid
        GRIDMARGIN = self.scale * 3
        
        #agent window size
        agentWIDTH = self.scale * 2
        agentHEIGHT = self.scale * 2
        
        if agentWIDTH * xblocks + MARGIN * (xblocks + 1) + GRIDMARGIN * 2 > (HEIGHT * yblocks + MARGIN * (yblocks + 1)) * ygrids + GRIDMARGIN * (ygrids):
            WINDOW_SIZE = [(WIDTH * xblocks + MARGIN * (xblocks + 1))* xgrids + GRIDMARGIN * (xgrids + 1) + agentWIDTH * xblocks + MARGIN * (xblocks + 1)
                           ,agentHEIGHT * yblocks + MARGIN * (yblocks + 1) + GRIDMARGIN * 2]
        else:
            WINDOW_SIZE = [(WIDTH * xblocks + MARGIN * (xblocks + 1))* xgrids + GRIDMARGIN * (xgrids + 1) + agentWIDTH * xblocks + MARGIN * (xblocks + 1)
                           ,(HEIGHT * yblocks + MARGIN * (yblocks + 1)) * ygrids + GRIDMARGIN * (ygrids)]
            
        
        # This will be the spacing between grid topleft corner
        XGRIDSPACING = WIDTH * xblocks + MARGIN * (xblocks + 1) + GRIDMARGIN
        YGRIDSPACING = HEIGHT * yblocks + MARGIN * (yblocks + 1) + GRIDMARGIN
        aXGRIDSPACING = agentWIDTH * xblocks + MARGIN * (xblocks + 1) + GRIDMARGIN
        aYGRIDSPACING = agentHEIGHT * yblocks + MARGIN * (yblocks + 1) + GRIDMARGIN
        
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
        grids.append([])
        x = GRIDMARGIN
        y = GRIDMARGIN * 2
        grids[0].append(Grid(xblocks,yblocks,x,y,screen,(agentWIDTH,agentHEIGHT,MARGIN,hsize),len(self.heatmaps)))
        
        for row in range(len(self.heatmaps)):
            # Add an empty array that will hold each cell
            # in this row
            grids.append([])
            for column in range(len(self.heatmaps[0])):
                x = GRIDMARGIN + (XGRIDSPACING * column) + aXGRIDSPACING
                y = GRIDMARGIN + (YGRIDSPACING * row)
                grids[row + 1].append(Grid(xblocks,yblocks,x,y,screen,(WIDTH,HEIGHT,MARGIN,hsize),len(self.heatmaps))) 
        """
        for x in range(len(self.moves)):
            all_sprites.add(Player(GRIDMARGIN * 0.5 + XGRIDSPACING * row,GRIDMARGIN * 0.5 + YGRIDSPACING * column))
        """
        
        #grids[0][0].addArrows(all_sprites.sprites())
        grids[0][0].addMoves(self.moves)
        
        font1 = pygame.font.Font('freesansbold.ttf',self.scale * 4)
        font2 = pygame.font.Font('freesansbold.ttf',self.scale * 2)
        text = []
        text.append(font1.render('Agents',True,GREEN,BLUE))
        
        
        """        
        print("heeats:")
        print(len(self.heatmaps))
        print(len(self.heatmaps[0]))
        """
        
        gridsUsed = []
        #updated
        grids[0][0].addMap(self.groundTruth)
        gridsUsed.append(grids[0][0])
        for x in range(len(self.heatmaps)):
            for y in range(len(self.heatmaps[0])):
                
                grids[x + 1][y].addMap(self.heatmaps[x][y] )
                gridsUsed.append(grids[x + 1][y])

                if len(self.datadisplayed) > y:
                    #this will add numbers to the agents
                    text.append(font2.render(self.datadisplayed[y] + " " + str(x),True,grids[0][0].colors[x],GREY))

        textRect = []
        
        for x in range(len(text)):
            textRect.append(text[x].get_rect())
            
        # Set title of screen
        if self.suppressOutput is 0:
            pygame.display.set_caption("Multi-robot Informative Sampling")
         
        # Loop until the user clicks the close button.
        done = False
         
        # Used to manage how fast the screen updates
        movecount = 0
        
        while not done:
            if self.suppressOutput is 0:
                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        done = True  # Flag that we are done so we exit this loop
    
            if movecount <= len(self.moves[0]):
                screen.fill(GREY)
                for idx , x in enumerate(gridsUsed):
                    x.move()
                    #all_sprites.update()
                    if self.suppressVideo is 0:
                        x.draw()
                        if len(textRect) >idx:
                            textRect[idx].bottomleft = (x.cornerx,x.cornery - self.scale // 2)
                            screen.blit(text[idx], textRect[idx])
                            pygame.image.save(screen,self.exportDirectory + "frame" + str(movecount) + ".png")
                        #all_sprites.draw(screen)
                            
            if movecount == len(self.moves[0]) - 1:
                screen.fill(GREY)
                self.suppressVideo = 0
                for x in grids:
                    for y in x:
                        if y.newmap is not None:
                            y.gridnumber = len(y.newmap) - 1
                            y.draw()
                            #pygame.image.save(screen,self.exportDirectory + "frame" + str(movecount + 1) + ".png")
                            #all_sprites.draw(screen)
                    
            if self.suppressOutput is 0:      
                pygame.display.flip()
            
            if movecount == len(self.moves[0]) - 1:
                #pygame.image.save(screen,self.exportDirectory + "out.png")
                pygame.quit()
                done = True
            
            movecount += 1
        
        print("Saving output.")
        if self.suppressVideo == 0:
            filenames = []
            for x in range(len(self.moves[0])):
                filenames.append(self.exportDirectory + "frame" + str(x) + ".png")
            images = []
            for filename in filenames:
                images.append(imageio.imread(filename))
            images.append(images[len(images) - 1])
            images.append(images[len(images) - 1])
            images.append(images[len(images) - 1])
            images.append(images[len(images) - 1])
            images.append(images[len(images) - 1])
            imageio.mimsave(self.exportDirectory + "anim.mp4", images , fps = 1)
            for filename in filenames:
                os.remove(filename)
                
        self.coverage()
 
        pygame.quit()
        
        
    def coverage(self):
        
        def truncate(f, n):
            s = '%.12f' % f
            i, p, d = s.partition('.')
            return '.'.join([i, (d+'0'*n)[:n]])
        
        print("Getting coverage...",end='')
        
        # Define some colors
        GREY = (100, 100, 100)
        GREEN = ( 50, 255, 50)
        BLUE = (25, 25, 150) 
        
        self.scale = self.scale * 2 
        # This sets the WIDTH and HEIGHT of each grid location and player
        WIDTH = self.scale
        HEIGHT = self.scale
        #history indent, previous location rectangle size
        hsize = self.scale
        
        #dimensions of grid
        xblocks = self.dimensions[0]
        yblocks = self.dimensions[1]
        xgrids = 2
        """
        if len(self.heatmaps) % xgrids is 0:
            ygrids = len(self.heatmaps) // xgrids
        else:
            ygrids = len(self.heatmaps) // xgrids + 1
        """
        ygrids = 1
        # This sets the margin between each cell
        MARGIN = self.scale // 11 + 1
        # This sets the margin between each grid
        GRIDMARGIN = self.scale * 3 // 2
        
        WINDOW_SIZE = [(WIDTH * xblocks + MARGIN * (xblocks + 1))* xgrids + GRIDMARGIN * (xgrids + 1.5)
                       ,(HEIGHT * yblocks + MARGIN * (yblocks + 1)) * ygrids + GRIDMARGIN * (ygrids + 1.5)]
        
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
                y = GRIDMARGIN * 2 + (YGRIDSPACING * column)
                grids[row].append(Grid(xblocks,yblocks,x,y,screen,(WIDTH,HEIGHT,MARGIN,hsize),len(self.heatmaps)))
        """
        for x in range(len(self.moves)):
            all_sprites.add(Player(GRIDMARGIN * 0.5 + XGRIDSPACING * row,GRIDMARGIN * 0.5 + YGRIDSPACING * column))
        """
        
        #grids[0][0].addArrows(all_sprites.sprites())
        grids[0][0].addMoves(self.moves)
        
        #figure out coverage
        visited = [([0]*xblocks) for i in range(yblocks)]
        
        for x in self.moves:
            for y in x:
                visited[y[1]][y[0]] = 1
                
        count = 0        
        for x in range(len(visited)):
            for y in range(len(visited[0])):
                if visited[x][y] == 1:
                    count = count + 1
        #updated
        temp = []
        temp.append(visited)
        grids[0][0].addMap(temp)
        
        #captions
        font = pygame.font.Font('freesansbold.ttf', int((0.05 * self.dimensions[0] + 0.4) * self.scale))
        text = []
        
        """
        #updated
        text.append(font.render(self.datadisplayed + " " + str(1),True,grids[0][0].colors[self.agentsTraced[0] - 1],GREY))
        text.append(font.render(self.datadisplayed + " " + str(2),True,grids[0][0].colors[self.agentsTraced[1] - 1],GREY))
        """
        
        textRect = []
            
        # Set title of screen
        if self.suppressOutput is 0:
            pygame.display.set_caption("Multi-robot Informative Sampling - Coverage")
         
        # Loop until the user clicks the close button.
        done = False
        
        screen.fill(GREY)  
        cm = grids[0][0].draw()
        bag = []
        
        for q in cm:
            for p in q:
                t = (p[0],p[1],p[2])
                if t not in bag:
                    bag.append(t)
        color1 = (bag[0][0] * 255 , bag[0][1] * 255 , bag[0][2] * 255)
        color2 = (bag[1][0] * 255 , bag[1][1] * 255 , bag[1][2] * 255)
        text.append(font.render("Coverage during episode" ,True,color1,color2))
        text.append(font.render(" Nodes covered: " + str(count) + " ",True,color1,color2))
        temp = truncate(count * 100/(xblocks * yblocks),2)
        text.append(font.render(" Percentage covered: " + str(temp) + " % ",True,color1,color2))
        temp = truncate(len(self.moves[0]) * len(self.moves) * 100/(xblocks * yblocks),2)
        text.append(font.render(" Best possible coverage: " + str(temp) + "% ",True,color1,color2))
        textRect.append(text[0].get_rect())
        textRect.append(text[1].get_rect())
        textRect.append(text[2].get_rect())
        textRect.append(text[3].get_rect())
        GRIDMARGIN = GRIDMARGIN * 2
        textRect[0].bottomleft = (grids[0][0].cornerx,grids[0][0].cornery-5)
        textRect[1].bottomleft = (grids[1][0].cornerx,grids[1][0].cornery+GRIDMARGIN)
        textRect[2].bottomleft = (grids[1][0].cornerx,grids[1][0].cornery+GRIDMARGIN * 2)
        textRect[3].bottomleft = (grids[1][0].cornerx,grids[1][0].cornery+GRIDMARGIN * 3)
        
        screen.blit(text[0], textRect[0])
        screen.blit(text[1], textRect[1])
        screen.blit(text[2], textRect[2])
        screen.blit(text[3], textRect[3])
        if self.suppressOutput is 0:
            pygame.display.flip()
        print("Saving output.")
        pygame.image.save(screen,self.exportDirectory + "coverage.png")
        
        while not done:
            if self.suppressOutput is 0:
                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        done = True
            else:
                done = True
 
 
        self.scale = self.scale / 2
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
        self.colors = [(116, 155, 254),(217, 143, 218),(195, 223, 232),(237, 177, 139),(235, 154, 254),(235, 132, 106),(123, 243, 158),(205, 223, 247),(199, 207, 179),(128, 254, 142)]
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
        dif = len(self.moves) - len(self.colors)
        for x in range(dif):
            color = (random.randrange(155) + 100,random.randrange(155) + 100,random.randrange(155) + 100)
            self.colors.append(color)
        #random.random(
                           
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
            #self.newmap[self.gridnumber] = self.newmap[self.gridnumber].flatten()
            color_matrix = 0
            self.newmap[self.gridnumber] = np.array(self.newmap[self.gridnumber]).reshape(-1,self.xblocks)
            if len(self.newmap) is not 1 or self.im is None:
                im = plt.imshow(self.newmap[self.gridnumber])
                color_matrix = im.cmap(im.norm(im.get_array()))
                plt.clf()
            
        for row in range(self.yblocks):
            for column in range(self.xblocks):
                #print(self.yblocks)
                #print(self.xblocks)
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
                    #if self.playerloc.index((row,column)) < self.agentsTraced:
                        #numbering for agents is below
                        #newtext = self.font.render(str(self.playerloc.index((row,column))),True,(0,0,0))
                        #add width/4 to help algn text
                        #self.screen.blit(newtext,(xdest + self.WIDTH//4,ydest))
                elif self.visited[row][column] != (0,0,0,0):
                    color = (self.visited[row][column][0],self.visited[row][column][1],self.visited[row][column][2])
                    s = pygame.Surface((self.hsize,self.hsize))  # the size of your rect
                    s.set_alpha(self.visited[row][column][3])                # alpha level
                    s.fill(color)           # this fills the entire surface
                    xdest = (self.MARGIN + self.WIDTH) * column + self.MARGIN + self.cornerx + (self.WIDTH - self.hsize)/2
                    ydest = (self.MARGIN + self.HEIGHT) * row + self.MARGIN + self.cornery + (self.WIDTH - self.hsize)/2
                    self.screen.blit(s, (xdest,ydest) )
            
        self.colorDecay()
        self.gridnumber += 1
        return color_matrix
        
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