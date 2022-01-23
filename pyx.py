#! /usr/bin/env python
"""Pyx

Version 1.0

Copyright (c) 2002 by Ken Seehof <kseehof@neuralintegrator.com>

Current version is at http://www.neuralintegrator.com/pyx

Open Source license: GPL (http://www.gnu.org/copyleft/gpl.html)
The author reserves the right to release new versions under a different license.

---------------------------------------------------------------

THIS PACKAGE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE
IMPLIED WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A
PARTICULAR PURPOSE.

---------------------------------------------------------------

Requirements:
Python 2.1 or higher (might work on older versions): http://www.python.org
PyGame: http://www.pygame.org
"""

# configuration parameters:
RandomTest = 0
FullScreenMode = 1
LongShadow = 1
FrameRate = 26 # frame rate (fps) = FrameRate + level*2
StartLives = 4
ExtraLife = 50000
SpyrxBarSpeed = 5 # spyrx bar expiration rate = SpyrxBarSpeed+level
FillScore = [0,5,10]
StartLevel = 1
NoSound = 0
ShowFPS = 0

if RandomTest:
    ExtraLife = 400
    StartLives = 16
    FrameRate = 10000

import random, os.path, sys
import array
import time
import string

#import basic pygame modules
import pygame, pygame.image, pygame.transform, pygame.sprite, pygame.draw
import pygame.mixer
from pygame.locals import *
#import crystal


def make_map(a,b):
    return dict([(a[i], b[i]) for i in range(len(a))])

shift_map = make_map(
    string.ascii_lowercase+'1234567890[]\';\\/,.`-=',
    string.ascii_uppercase+'!@#$%^&*(){}":|?<>~_+')
      
def load_digits(file, (width,height)):
    'chops a bitmap and returns a list of 10 digit surfaces and a comma if there is extra horizontal width'
    image_block = pygame.image.load(file).convert_alpha()
    images = []

    comma_width = image_block.get_width() - width*10

    for j in range(11):
        if j==10:
            if comma_width>0:
                surface = image_block.subsurface((j*width,0,comma_width,height))
                images.append(surface)
        else:
            surface = image_block.subsurface((j*width,0,width,height))
            images.append(surface)

    return images

def blit_number(surface, digit_images, (width,height), (x,y), ndigits, n, padspace=1):
    "blits zero or space padded number, no commas (don't forget to erase first)"
    s = ('%%0%dd'%ndigits)%n # n zero padded to ndigits
    for i in range(len(s)):
        if not (padspace and s[i]=='0' and i<len(s)-1):
            surface.blit(digit_images[ord(s[i])-ord('0')], (x+i*width,y))
            padspace=0

def load_ascii(file, (width,height)):
    'chops a bitmap and returns a dictionary of images indexed by character'
    ascii_image_block = pygame.image.load(file).convert_alpha()
    images = {}

    for j in range(95):
        surface = ascii_image_block.subsurface((j*width,0,width,height))
        images[chr(32+j)] = surface

    return images

def blit_text(surface, ascii_images, (width,height), (x,y), s):
    "blits text using ascii_images"
    for i in range(len(s)):
        surface.blit(ascii_images[s[i]], (x+i*width,y))


sound_dict = {}

def load_sounds():
    'loads all wav files in the sounds directory'
    if not pygame.mixer:
        print 'no mixer'
        return

    files = os.listdir('sounds')

    for file in files:
        sound_name, ext = os.path.splitext(file)
        if ext == '.wav':
            sound_dict[sound_name] = pygame.mixer.Sound(os.path.join('sounds', file))

class dummy_sound:
    def play(*args):
        print 'dummy'

def get_sound(sound_name):
    if NoSound:
        return dummy_sound()

    if sound_name in sound_dict:
        return sound_dict[sound_name]
    else:
        print 'unable to play sound "%s"' % sound_name
        return dummy_sound()


def unit_vector(p0,p1):
    'returns a king-move vector one space (in one of 8 directions) from p0 to p1'
    x0,y0 = p0; x1,y1 = p1
    ux = cmp(x1-x0, 0)
    uy = cmp(y1-y0, 0)

    if abs(x1-x0) > 2*abs(y1-y0):
        return (ux,0)
    
    if abs(y1-y0) > 2*abs(x1-x0):
        return (0, uy)

    return (ux,uy)

def d2(p0,p1):
    'distance squared from p0 to p1'
    x0,y0 = p0; x1,y1 = p1
    return (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0)


left_turn = {
    (0,1):(1,0),
    (1,0):(0,-1),
    (0,-1):(-1,0),
    (-1,0):(0,1),
    }

right_turn = {
    (0,1):(-1,0),
    (1,0):(0,1),
    (0,-1):(1,0),
    (-1,0):(0,-1),
    }

class Spryte(pygame.sprite.Sprite):
    '''Sprite customized for use in a Pyx Grid.  Position is in grid coordinates.'''
    def __init__(self, container, grid, position):
        pygame.sprite.Sprite.__init__(self, container)
        self.container = container
        self.grid = grid
        self.position = position


class Spyrx(Spryte):
    def __init__(self, container, grid, position, dir):
        Spryte.__init__(self, container, grid, position)
        image_block = pygame.image.load('spyrx.png').convert_alpha()
        self.images = []
        self.dir = dir
        if dir[0] == 1: # clockwise
            self.turn = [left_turn, right_turn]
        else: # counter-clockwise
            self.turn = [right_turn, left_turn]

        for i in range(8):
            for j in range(8):
                #surface = pygame.Surface((16,16)).convert_alpha()
                #surface.blit(image_block.subsurface((i*16,j*16,16,16)),(0,0))
                surface = image_block.subsurface((i*16,j*16,16,16))
                self.images.append(surface)
                
        x0,y0 = self.grid.offset
        x,y = self.position
        self.image = self.images[(x/2+y/2)%len(self.images)]
        self.rect = (x0+x*3-6, y0+y*3-6, 16, 16)

    def update(self):
        x0,y0 = self.grid.offset
        x,y = self.position
        dx,dy = self.dir
        g = self.grid

        if g.cell_wall(x,y):
            if not g.cell_wall(x+dx,y+dy):
                if g.cell_wall(x+dy,y+dx):
                    dx,dy = dy,dx
                elif g.cell_wall(x-dx,y-dy):
                    dx,dy = -dy,-dx
                else:
                    dx,dy = -dx,-dy
            x+=dx; y+=dy
        else:
            if g.cell_wall(x+dx,y+dy):
                x+=dx; y+=dy
                dx,dy = self.turn[0][(dx,dy)]
            else:
                g.add_crumb(x,y)
                best = (100000000,(1,0))
                for ddx,ddy in [self.turn[0][(dx,dy)], (dx,dy), self.turn[1][(dx,dy)]]:
                    if g.cell_traceable(x+ddx,y+ddy):
                        c = g.get_crumbs(x+ddx,y+ddy)
                        if c < best[0]:
                            best = (c,(ddx,ddy))
                dx,dy = best[1]
                x+=dx; y+=dy

        if (x,y)==self.grid.player.position or self.position==self.grid.player.position:
            self.grid.player.explode()

        self.image = self.images[(x/2+y/2)%len(self.images)]
        self.rect = (x0+x*3-6, y0+y*3-6, 16, 16)
        self.position = x,y
        self.dir = dx,dy

class SpyrxBar:
    def __init__(self, grid, speed):
        self.grid = grid
        self.speed = speed
        self.width = grid.width
        self.center = (self.width-2)/2
        self.reset()

    def reset(self, kill=0, speed=-1):
        get_sound('create_spyrx').play()

        self.a = 0
        for i in range(2, self.width-3):
            self.grid.grid_set(i,0,8)
        self.pos = (self.width-6)/2

        if kill:
            for spyrx in self.grid.spyrx:
                spyrx.kill()
            self.grid.spyrx = []

        if speed>-1:
            self.speed = speed
            
        if speed!=0:
            self.grid.spyrx.append(Spyrx(self.grid.sprites, self.grid, (self.center, 2),(1,0)))
            self.grid.spyrx.append(Spyrx(self.grid.sprites, self.grid, (self.center, 2),(-1,0)))

    def set_speed(self,speed):
        'set bar speed.  Max 100 = one unit per frame'
        self.speed = speed

    def update(self):
        if self.grid.player.game_over:
            return

        self.a += self.speed
        if self.a > 100:
            self.a -= 100
            self.grid.grid_set(self.center-self.pos, 0, 3)
            self.grid.grid_set(self.center+self.pos, 0, 3)
            self.pos-=1
            if self.pos < 0:
                self.reset()


class Fuse(Spryte):
    def __init__(self, player):
        Spryte.__init__(self, player.container, player.grid, player.position)
        self.player = player
        image_block = pygame.image.load('fuse.png').convert_alpha()
        self.images = []
        self.dir = (0,0)
        self.turn = [left_turn, right_turn]

        for i in range(8):
            for j in range(8):
                surface = image_block.subsurface((i*12,j*12,12,12))
                self.images.append(surface)
                
        x0,y0 = self.grid.offset
        x,y = self.position
        self.image = self.images[(x/2+y/2)%len(self.images)]
        self.rect = (x0+x*3-4, y0+y*3-4, 12, 12)

    def update(self):
        p = self.player
        if p.drawing and p.dir==(0,0) and not p.new and not p.dead:
            x0,y0 = self.grid.offset
            x,y = self.position
            dx,dy = self.dir
            g = self.player.grid

            if self.countdown:
                self.countdown -= 1
            else:
                if not g.cell_draw(x+dx,y+dy):
                    if g.cell_draw(x+dy,y+dx):
                        dx,dy = dy,dx
                    elif g.cell_draw(x-dx,y-dy):
                        dx,dy = -dy,-dx
                    else:
                        self.player.explode()
                        dx,dy = -dx,-dy # turn around if explosion is disabled
                x+=dx; y+=dy

                if (x,y)==self.grid.player.position or self.position==self.grid.player.position:
                    self.player.explode()

            self.image = self.images[1+(x/2+y/2+self.countdown)%(len(self.images)-1)]
            self.rect = (x0+x*3-4, y0+y*3-4, 12, 12)
            self.position = x,y
            self.dir = dx,dy
        else:
            self.image = self.images[0]

class Player(Spryte):
    '''The Player sprite, complete with drawing logic.'''
    key_list = [
            [K_UP],
            [K_RIGHT],
            [K_DOWN],
            [K_LEFT],
        ]
    dir_list = [
            (0,-1),
            (1,0),
            (0,1),
            (-1,0),
        ]
    
    def __init__(self, container, grid, position):
        Spryte.__init__(self, container, grid, position)
        self.name = '' # for high score list
        self.grid = grid
        self.start_position = position
        self.image = self.liveimage = pygame.image.load('player.png').convert_alpha()
        self.cloaked_image = pygame.image.load('player_cloaked.png').convert_alpha()
        self.explode_images = [pygame.image.load('exp%d.png'%i).convert_alpha() for i in range(7)]
        self.new_images = [pygame.image.load('new%d.png'%i).convert_alpha() for i in range(10)]
        self.fuse = Fuse(self)
        self.reset(0)

    def reset(self, start=1):
        self.position = self.start_position
        self.drawing =0
        self.partial = (0,0)
        self.score = 0
        self.score_dirty = 1
        self.dir_preference = [0,1,2,3] # optimal keyboard control with multiple keys held
        self.explosion = 0
        self.dead = 0
        self.bonus = 0
        self.bonus_pause = 0
        self.levelling = 0
        self.lives = StartLives-1 # excluding current life
        self.lives_dirty = 1
        self.next_extra_life = ExtraLife
        
        if start:
            self.new = 1
            self.game_over = 0
        else:
            self.new = 0
            self.game_over = -1
            
        

    def add_score(self,n):
        self.score += n
        self.score_dirty = 1
        while self.score >= self.next_extra_life:
            get_sound('extra').play()
            self.lives += 1
            self.next_extra_life += ExtraLife
            self.lives_dirty = 1

    def calc_dir(self, keystate):
        'convert keystate to list of selected direction idexes in order of preference, and draw flag'
        draw=0
        if keystate[K_LCTRL]: draw = 1 # fast draw
        if keystate[K_RCTRL]: draw = 1 
        if keystate[K_LSHIFT]: draw = 2 # slow draw
        if keystate[K_RSHIFT]: draw = 2
        
        dirlist = []
        for i in self.dir_preference:
            for k in self.key_list[i]:
                if keystate[k]:
                    dirlist.append(i)
                    break

        return dirlist, draw

    def prefer_dir(self, i):
        self.dir_preference.remove(i)
        self.dir_preference.insert(0,i)
        i = (i+2)%4
        self.dir_preference.remove(i)
        self.dir_preference.append(i)

    def unprefer_dir(self, i):
        self.prefer_dir((i+2)%4)

    def update(self):
        'Handle player action for one frame'
        x,y = self.position
        g = self.grid

        x0,y0 = self.grid.offset

        if self.game_over or self.levelling:
            self.rect = (x0+x*3-6, y0+y*3-6, 15, 15)
            return

        if self.dead:
            if self.explosion < len(self.explode_images):
                self.image = self.explode_images[int(self.explosion)]
                self.explosion += .25
            else:
                self.explosion = 0
                x,y = self.position
                x0,y0 = self.grid.offset
                self.rect = (x0+x*3-6, y0+y*3-6, 15, 15)

                if self.lives > 0:
                    self.lives -= 1
                    self.game_over = 0
                    self.lives_dirty = 1
                    self.dead = 0
                    self.new = 1
                    self.grid.spyrx_bar.reset(1)
                else:
                    get_sound('gameover').play()
                    self.lives_dirty = 1
                    self.game_over = 1
            return

        if self.new:
            if self.explosion < len(self.new_images):
                self.image = self.new_images[int(self.explosion)]
                self.explosion += .25
            else:
                self.explosion = 0
                self.new = 0
                x,y = self.position
                x0,y0 = self.grid.offset
                self.rect = (x0+x*3-6, y0+y*3-6, 15, 15)
                self.image = self.liveimage

        if (x+y)&1 == 0:
            # even point: can change dir
            keystate = pygame.key.get_pressed()
            dirlist,draw = self.calc_dir(keystate)
            
            if RandomTest:
                i = random.randint(0,3)
                self.prefer_dir(i)
                dirlist = self.dir_preference
                draw=random.randint(0,2)

            if self.drawing:
                for i in dirlist:
                    dx,dy = self.dir_list[i]
                    
                    if g.cell_empty(x+dx*2, y+dy*2) or g.cell_wall(x+dx*2, y+dy*2) and (x+dx*2, y+dy*2) != self.launch_point:
                        # direction succeeded: move to beginning of preference list
                        self.prefer_dir(i)
                        self.dir = dx,dy

                        if self.drawing == 2 and self.partial == (0,0):
                            self.partial = self.dir
                        else:
                            self.partial = (0,0)
                            x+=dx; y+=dy
                            g.grid_set(x,y,4+self.drawing)

                        self.position = x,y
                        break
                    else:
                        # direction blocked: move direction to end of preference list
                        self.unprefer_dir(i)
                else:
                    self.dir = (0,0)
            else:
                for i in dirlist:
                    dx,dy = self.dir_list[i]
                    if g.cell_wall(x+dx, y+dy):
                        # direction succeeded: move to beginning of preference list
                        self.prefer_dir(i)
                        self.dir = dx,dy
                        x+=dx; y+=dy
                        self.position = x,y
                        break
                    elif draw and g.cell_empty(x+dx, y+dy) and (g.cell_empty(x+dx*2, y+dy*2) or g.cell_wall(x+dx*2, y+dy*2)):
                        # launch
                        # direction succeeded: move to beginning of preference list
                        self.unprefer_dir(i)
                        self.launch_point = (x,y)
                        self.fuse.dir = self.dir = dx,dy
                        self.fuse.position =  (x+dx,y+dy)
                        self.fuse.countdown = 20
                        x+=dx; y+=dy
                        self.position = x,y
                        self.drawing = draw
                        g.grid_set(x,y,4+draw)
                        break
                    else:
                        # direction blocked: move direction to end of preference list
                        self.unprefer_dir(i)
                else:
                    self.dir = (0,0)
        else:
            # odd point
            dx,dy = self.dir

            if self.drawing:
                if g.cell_wall(x+dx, y+dy):
                    x+=dx; y+=dy
                    # landing
                    # corner points pa,pb
                    pa = x-dx-dy, y-dy-dx
                    pb = x-dx+dy, y-dy+dx
                    
                    if g.cell_wall(x-dy, y-dx):
                        da = -dy,-dx
                    else:
                        da = dx,dy

                    if g.cell_wall(x+dy, y+dx):
                        db = +dy,+dx
                    else:
                        db = dx,dy

                    g.trace((x,y), self.launch_point, (-dx,-dy), 4+self.drawing, 10)
                    #excluding da
                    g.trace((x,y), self.launch_point, da, 7, 11)

                    get_sound('fill').play()

                    test_point = g.pyx.p[0]
                    if g.seek(pa, test_point, [10,7]):
                        n = g.fill(pb, [10,7], self.drawing)
                        self.add_score(FillScore[self.drawing] * n)
                        g.current_filling = n
                        g.trace((x,y), self.launch_point, da, 11, 7) # rewall
                        g.trace((x,y), self.launch_point, db, 7, 4) # bury
                        g.trace((x,y), self.launch_point, (-dx,-dy), 10, 7) # solidify
                    else:
                        xa,ya = self.launch_point
                        g.grid_set(xa,ya, 11)
                        g.grid_set(x,y, 11)
                        n = g.fill(pa, [10,11], self.drawing)
                        g.current_filling = n
                        self.add_score(FillScore[self.drawing] * n)
                        g.grid_set(xa,ya, 7)
                        g.grid_set(x,y, 7)
                        g.trace((x,y), self.launch_point, da, 11, 4) # bury
                        g.trace((x,y), self.launch_point, (-dx,-dy), 10, 7) # solidify

                    if g.percent >= 75:
                        g.spyrx_bar.reset(1,0)
                        self.levelling = 1
                        self.bonus = (g.percent-75)*1000
                        self.bonus_pause = 60
                        
                    if not g.cell_empty(*g.pyx.p[1]):
                        g.pyx.p[1] = g.pyx.p[0]
                    
                    self.drawing = 0
                    self.launch_point = None
                else:
                    if self.drawing == 2 and self.partial == (0,0):
                        self.partial = self.dir
                    else:
                        self.partial = (0,0)
                        x+=dx; y+=dy
                        g.grid_set(x,y,4+self.drawing)
            else:
                x+=dx; y+=dy

        self.position = x,y
        x0,y0 = self.grid.offset
        xp0,yp0 = self.partial # smooth out slow draw
        self.rect = (x0+x*3+xp0-6, y0+y*3+yp0-6, 15, 15)

    def explode(self):
        "call this when it's time to die (currently not very destructive!)"
        if self.dead or self.new or self.game_over or self.levelling:
            return

        get_sound('death').play()
        
        if self.drawing:
            x,y=self.position
            dx,dy = self.dir
            self.grid.grid_set(x,y,0)
            self.grid.trace(self.position, self.launch_point, (-dx,-dy), 4+self.drawing, 0)
            self.position = self.launch_point
            self.drawing = 0
            self.launch_point = None

        self.dead = 1
        

    def find_odd_space(self):
        'returns an adjacent empty odd/odd point'
        x,y = self.position
        g = self.grid

        if (x+y)&1:
            dlist = [(1,0), (-1,0), (0,-1), (0,1)]
        else:
            dlist = [(1,1), (-1,1), (1,-1), (-1,-1)]

        for (dx,dy) in dlist:
            if g.cell_empty(x+dx, y+dy):
                return (x+dx, y+dy)

        raise ValueError('find_odd_space failed')

class Pyx:
    '''The bundle of lines that buzzes around the grid.
    Pyx endpoints always move in knight moves at constant speed. (pleasant visual effect)
    Knight moves are split into two phases (e.g. forward-left, forward)
    Endpoints are always odd/odd points.
    There are a couple A/I features to chase around walls and avoid straddling.'''
    
    unit_vectors = [(1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1)] # king moves
    uvec_index = dict([(unit_vectors[i],i) for i in range(8)])
    directions = [(1,-2), (2,-1), (2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2)] # knight moves
    dir_index = dict([(directions[i],i) for i in range(8)])

    # directional knight move operators    
    dir_reflect_x = dict([((dx,dy),(-dx,dy)) for dx,dy in directions])
    dir_reflect_y = dict([((dx,dy),(dx,-dy)) for dx,dy in directions])
    dir_clockwise = dict([(d,directions[(dir_index[d]+1)%8]) for d in directions])
    dir_cclockwise = dict([(d,directions[(dir_index[d]+7)%8]) for d in directions])
    dir_approach = {}
    for i in range(8):
        for j in range(4):
            dir_approach[(directions[i], unit_vectors[(i+j)%8])] = dir_clockwise[directions[i]]
            dir_approach[(directions[i], unit_vectors[(i+j+4)%8])] = dir_cclockwise[directions[i]]

    dir_steps = {} # knight move as three unit steps
    for i in range(8):
        dx,dy = directions[i]
        ux,uy = cmp(dx,0), cmp(dy,0)
        if abs(dx) > abs(dy):
            dir_steps[(dx,dy)] = ((ux,0),(0,uy),(ux,0))
        else:
            dir_steps[(dx,dy)] = ((0,uy),(ux,0),(0,uy))

    def __init__(self, grid):
        self.grid = grid
        self.p = [(55,55), (57,59)] #position
        self.d = [(1,2), (2,1)] #direction (always knights move)
        self.history = []
        self.phase = 0 # half of knights move (choices made in phase 0, minor axis offset in phase 1)
        self.reset()

    def reset(self):
        level = self.grid.level
        self.strategies = {}
        # probability distribution of strategies
        self.strategies[self.strategy_momentum] = max(0,40-level*2)
        self.strategies[self.strategy_random] = 20
        self.strategies[self.strategy_shrink] = 20 #d2(*self.p)/200
        self.strategies[self.strategy_chase] = level

    def pscan(self, p0,p1):
        'search for line crossing from odd points p0 to p1; return first crossing point and direction'
        g = self.grid
        x,y = x0,y0 = p0
        x1,y1 = p1
        ux,uy = cmp(x1,x0), cmp(y1,y0)
        adx, ady = abs(x1-x0)+1, abs(y1-y0)+1

        while (x,y) != (x1,y1):
            if abs((x1-x)*ady) > abs((y1-y)*adx):
                if not g.cell_empty(x+ux,y):
                    return ((x+ux,y), (ux,0))
                else:
                    x += 2*ux
            else:
                if not g.cell_empty(x,y+uy):
                    return ((x,y+uy), (0, uy))
                else:
                    y += 2*uy

        return None

    def trace_race(self, p0,d0, p1,d1):
        '''Trace from p0 and p1 clockwise (d0 and d1 respectively) and see who finds the other.
        Return directions from p0 and p1 towards each other.'''
        g = self.grid
        x0,y0=p0; x1,y1=p1
        dx0,dy0=d0; dx1,dy1=d1
        while 1:
            if not g.cell_wall(x0+dx0, y0+dy0):
                if g.cell_wall(x0+dy0, y0+dx0):
                    dx0,dy0 = dy0,dx0
                elif g.cell_wall(x0-dy0, y0-dx0):
                    dx0,dy0 = -dy0,-dx0
                else:
                    raise ValueError('trace_race failed')
            x0+=dx0; y0+=dy0
            if (x0,y0) == p1:
                return d0, (-d1[0], -d1[1])

            if not g.cell_wall(x1+dx1, y1+dy1):
                if g.cell_wall(x1+dy1, y1+dx1):
                    dx1,dy1 = dy1,dx1
                elif g.cell_wall(x1-dy1, y1-dx1):
                    dx1,dy1 = -dy1,-dx1
                else:
                    raise ValueError('trace_race failed')
            x1+=dx1; y1+=dy1
            if (x1,y1) == p0:
                return (-d0[0], -d0[1]), d1

    def strategy_momentum(self, i):
        pass # keep going same direction

    def strategy_random(self, i):
        if random.randint(0,1):
            self.d[i] = self.dir_clockwise[self.d[i]]
        else:
            self.d[i] = self.dir_cclockwise[self.d[i]]

    def strategy_shrink(self, i):
        u = unit_vector(self.p[i], self.p[1-i])
        if u!=(0,0):
            self.d[i] = self.dir_approach[(self.d[i], u)]

    def strategy_chase(self, i):
        g = self.grid
        u = unit_vector(self.p[i], self.grid.player.position)
        if u==(0,0):
            return
        
        self.d[i] = self.dir_approach[(self.d[i], u)]

        p_empty = self.grid.player.find_odd_space()
        hit0 = self.pscan(self.p[i], p_empty)
        hit1 = self.pscan(p_empty, self.p[i])
        if hit0 and hit1:
            p0,u0 = hit0
            p1,u1 = hit1
            if g.cell_wall(*p0) and g.cell_wall(*p1):
                d0,d1 = self.trace_race(p0,right_turn[u0], p1,right_turn[u1])
                self.d[i] = self.dir_approach[(self.d[i], d0)]
 
    def select_strategy(self):
        stlist = self.strategies.items()
        s=0
        for st,p in stlist:
            s+= p
        n = random.randint(0,s-1)
        for st,p in stlist:
            if n<p: return st
            n-= p

    def update(self):
        g = self.grid
        if g.player.levelling:
            return
        
        pg = []
        xoff, yoff = g.offset
        
        for i in range(2):
            if self.phase == 0:
                # phase 0: choose direction, take first two steps and adjust minor axis
                strategy = self.select_strategy()
                strategy(i)

                x,y = self.p[i]
                steps = self.dir_steps[self.d[i]]
                
                # step 0
                dx,dy = steps[0]
                if g.cell_wall(x+dx, y+dy):
                    if (dx):
                        self.d[i] = self.dir_reflect_x[self.d[i]]
                    else:
                        self.d[i] = self.dir_reflect_y[self.d[i]]
                else:
                    x+=2*dx; y+=2*dy

                # step 1
                dx,dy = steps[1]
                if g.cell_wall(x+dx, y+dy):
                    if (dx):
                        self.d[i] = self.dir_reflect_x[self.d[i]]
                    else:
                        self.d[i] = self.dir_reflect_y[self.d[i]]
                else:
                    x+=2*dx; y+=2*dy

                pg.append((xoff+(x-dx)*3, yoff+(y-dy)*3))
                self.p[i] = (x,y)
            else:
                # phase 1: take last step
                x,y = self.p[i]
                steps = self.dir_steps[self.d[i]]

                # step 2
                dx,dy = steps[2]
                if g.cell_wall(x+dx, y+dy):
                    if (dx):
                        self.d[i] = self.dir_reflect_x[self.d[i]]
                    else:
                        self.d[i] = self.dir_reflect_y[self.d[i]]
                else:
                    x+=2*dx; y+=2*dy

                pg.append((xoff+x*3, yoff+y*3))
                self.p[i] = (x,y)

        hit0 = self.pscan(self.p[0], self.p[1])
        hit1 = self.pscan(self.p[1], self.p[0])

        if hit0:
            p0,u0 = hit0
            if g.cell_draw(*p0):
                g.player.explode()
            else:
                if hit1:
                    p1,u1 = hit1
                    if g.cell_draw(*p1):
                        g.player.explode()
                    else:
                        d0,d1 = self.trace_race(p0,right_turn[u0], p1,right_turn[u1])
                        self.d[0] = self.dir_approach[(self.d[0], d0)]
                        self.d[1] = self.dir_approach[(self.d[1], d1)]

        self.history.append(pg)
        self.history = self.history[-8:]
        self.phase ^= 1

    def paint(self, surface):
        for (p0,p1) in self.history:
            pygame.draw.line(surface, (255,0,0), p0,p1,2)


class Grid:
    '''The Pyx Grid.
    Edge vertexes are always on even/even points.
    Odd/odd points are never walls, and are convenient for region calculations.
    Extra bottom and right edges are for extra shadow width for sprites
    
    3 3 3 8 8 8 8 8 3 3 3 3
    3 3 3 3 3 3 3 3 3 3 3 3
    3 3 7 7 7 7 7 7 7 3 3 3
    3 3 7 0 0 0 0 0 7 3 3 3
    3 3 7 1 0 0 7 7 7 3 3 3
    3 3 7 0 0 0 7 2 4 3 3 3
    3 3 7 7 7 7 7 4 4 3 3 3
    3 3 3 3 3 3 3 3 3 3 3 3
    3 3 3 3 3 3 3 3 3 3 3 3
    3 3 3 3 3 3 3 3 3 3 3 3

    EMPTY = 0
    FILL1 = 1
    FILL2 = 2
    OUTSIDE = 3
    BWALL = 4
    DRAW1 = 5
    DRAW2 = 6
    WALL = 7
    SPYRXBAR = 8
    marked draw = 10
    marked wall = 11
    '''

    def __init__(self, grid_size):
        # boolean set membership functions
        self.cell_drawable = self.vmap(0,1,2,3,4,5,6,7,8)        
        self.cell_casts_shadow = self.vmap(5,6,7,8,11)
        self.cell_outside = self.vmap(1,2,3,4,8)
        self.cell_inside = self.vmap(0,5,6)
        self.cell_draw = self.vmap(5,6)
        self.cell_wall = self.vmap(7)
        self.cell_traceable = self.vmap(4,7)
        self.cell_empty = self.vmap(0)

        self.width, self.height = w,h = grid_size
        self.surface = pygame.Surface((w*3, h*3)).convert()
        self.total_area = (w-6)*(h-6)/4
        self.spyrx = []

        if w%2==1 or h%2==1 or w<7 or h<7:
            raise ValueError('grid dimensions must be even numbers > 5')
        
        self.level = StartLevel
        self.level_dirty=1
        
        # Cell colors: ((light rgb), (dark rgb))
        self.cell_color = [
            ((0,0,0),(0,0,0)), # empty
            ((0,0,180),(0,0,70)), # fill1
            ((160,0,0),(60,0,0)), # fill2 (slow)
            ((74,54,94),(30,20,45)), # outside (match background color)
            ((160,120,170),(60,40,70)), # bwall
            ((0,60,255),(0,0,200)), # draw1
            ((255,30,0),(180,0,0)), # draw2 (slow)
            ((255,255,255),(120,120,120)), # wall
            ((255,80,40),(200,0,0)), # sparxbar
            ]

        # Shadow templates:
        # Each byte is a color selector for the corresponding pixel (there are 9 pixels).
        # Each bit determines if the pixel is shaded (or not hilited) according to 3 neibor pixels.
        # The bit position (0-7) is the raised state of the adjacent pixels (1=left, 2=top, 4=corner)
        # E.g.: bit 5 of the byte selects the color when the left(1) and corner(4) cells are raised.
        
        self.t_long_shadow = self.parse_template('F0 FC CC FA F0 CC AA AA 00')
        self.t_short_shadow = self.parse_template('F0 CC CC AA 00 00 AA 00 00')
        self.t_raised = self.parse_template('00 CC CC AA FF FF AA FF FF')

        # Predraw 3x3 blits from shadow templates
        self.gridblit = []
        for v in range(len(self.cell_color)):
            colors = self.cell_color[v]

            if self.cell_casts_shadow(v):
                template = self.t_raised
            else:
                if LongShadow:
                    template = self.t_long_shadow
                else:
                    template = self.t_short_shadow

            for bx in range(8):
                b = 1<<bx
                gb = pygame.Surface((3,3)).convert()
                self.gridblit.append(gb)

                gb.set_at((0,0), colors[template[0]&b!=0])
                gb.set_at((1,0), colors[template[1]&b!=0])
                gb.set_at((2,0), colors[template[2]&b!=0])
                gb.set_at((0,1), colors[template[3]&b!=0])
                gb.set_at((1,1), colors[template[4]&b!=0])
                gb.set_at((2,1), colors[template[5]&b!=0])
                gb.set_at((0,2), colors[template[6]&b!=0])
                gb.set_at((1,2), colors[template[7]&b!=0])
                gb.set_at((2,2), colors[template[8]&b!=0])

        self.reset(1)

    def reset(self, reset_game=0):
        w,h = self.width, self.height
        self.grid = array.array('b', [0]*w*(h+2))
        self.crumbs = array.array('b', [0]*w*(h+2))
        self.dirty = {}
        self.qdirty = [] # "threaded" dirty pixels (do a few at a time in random order)
        self.surface.fill(self.cell_color[0][0])
        self.filled_area = 0
        self.current_filling = 0
        self.percent = 0
        
        #outside edge
        for x in range(0,w):
            self.grid_set(x, 0, 3)
            self.grid_set(x, 1, 3)
            self.grid_set(x, h-3, 3)
            self.grid_set(x, h-2, 3)
            self.grid_set(x, h-1, 3)
        for y in range(0,h):
            self.grid_set(0, y, 3)
            self.grid_set(1, y, 3)
            self.grid_set(w-3, y, 3)
            self.grid_set(w-2, y, 3)
            self.grid_set(w-1, y, 3)

        #wall
        for x in range(2,w-3):
            self.grid_set(x, 2, 7)
            self.grid_set(x, h-4, 7)
        for y in range(2,h-3):
            self.grid_set(2, y, 7)
            self.grid_set(w-4, y, 7)

        if reset_game:
            self.level = StartLevel
            self.level_dirty=1


    def vmap(self, *vset):
        'return a boolean function for membership in vset of cell (x,y) or value x:  f(x,y) or f(x)'
        a = [i in vset for i in range(16)]
        
        def cell_f(x,y=None,self=self):
            if y is None:
                return a[x]
            else:
                if y>=0:
                    return a[self.grid[self.width*y + x]]
                else:
                    return 0

        return cell_f

    def parse_template(self, template):
        'parse a shadow/hilite template from hex string'
        return [int(s, 16) for s in template.split()]

    def trace(self, p0, p1, dir, v0, v1):
        'change grid values v0 to v1, along path from p0 to p1 (both exclusive), starting in direction dir; return final dir'
        x,y = p0
        dx,dy = dir

        turn={
            (0,0):(0,1),
            (1,0):(0,1),
            (0,1):(-1,0),
            (-1,0):(0,-1),
            (0,-1):(1,0),
            }

        tc=0 # dbg
        if self.cell_drawable(v1):
            grid_set = self.grid_set
        else:
            grid_set = self.grid_set_nodraw
        
        while 1:
            if p1 == (x+dx, y+dy): return (dx,dy)

            if self.grid_get(x+dx, y+dy) != v0:
                (dx,dy) = turn[(dx,dy)]
                tc+=1
                if tc>4:
                    # Trace Failure: print some diagnostics
                    print 'p0,p1,dir, v0,v1 =', (p0, p1, dir, v0, v1)
                    print 'x,y,dx,dy =', (x,y,dx,dy)
                    for ey in range(-2,3):
                        print
                        for ex in range(-2,3):
                            print self.grid_get(x+ex, y+ey),
                    raise ValueError('trace failed at %s' % `(x,y)`)
                continue

            tc=0

            x+=dx; y+=dy
            grid_set(x,y,v1)


    def seek(self, p0, p1, vlist):
        'returns true if odd/odd points p0 and p1 are in the same region with a border defined by vlist (list of cell types)'
        vf = self.vmap(*vlist)
        x0,y0 = p0
        x1,y1 = p1
        c=0

        if x1>x0:
            for x in range(x0+1, x1):
                if vf(x,y0): c+=1
        else:            
            for x in range(x1+1, x0):
                if vf(x,y0): c+=1

        if y1>y0:
            for y in range(y0+1, y1):
                if vf(x1,y): c+=1
        else:            
            for y in range(y1+1, y0):
                if vf(x1,y): c+=1

        return not (c&1)

    def fill(self, p0, vlist, v, invert=0):
        'fills a region, bordered by vlist and containing odd point p0, with v'
        vf = self.vmap(*vlist)
        y_filling = self.seek(p0, (3,3), vlist)
        if invert: y_filling = not y_filling
        area = 0
        
        for y in range(3, self.height-3, 2):
            filling = y_filling
            for x in range(3, self.width-3):
                if vf(x,y):
                    filling = not filling
                else:
                    if filling:
                        #self.grid_set(x,y,v) # hard coded for speed
                        self.grid[self.width*y + x] = v
                        self.qdirty.append((x,y))
                      
                        area += (x&1) # count odd points only
                        if not vf(x,y+1):
                            #self.grid_set(x,y+1,v) # hard coded for speed
                            self.grid[self.width*(y+1) + x] = v
                            self.qdirty.append((x,y+1))
            if vf(3, y+1):
                y_filling = not y_filling

        self.filled_area += area
        self.percent = 100*self.filled_area/self.total_area
        return area

    def grid_set(self, x,y,v):
        'set a grid cell value and mark for painting'
        pv = self.grid[self.width*y + x]
        self.grid[self.width*y + x] = v
        if self.cell_drawable(v):
            self.dirty[(x,y)] = 1
            if self.cell_casts_shadow(v) != self.cell_casts_shadow(pv):
                self.dirty[(x+1,y)] = 1
                self.dirty[(x,y+1)] = 1
                self.dirty[(x+1,y+1)] = 1

    def grid_set_nodraw(self, x,y,v):
        'set a grid cell value and mark for painting'
        self.grid[self.width*y + x] = v

    def grid_get(self, x,y):
        'return the cell value at grid point x,y'
        return self.grid[self.width*y + x]

    def add_crumb(self, x,y):
        'add a crumb (used by spyrx to excape after being buried)'
        self.crumbs[self.width*y + x] += 1

    def get_crumbs(self, x,y):
        'count crumbs (used by spyrx to excape after being buried)'
        return self.crumbs[self.width*y + x]

    def paint(self):
        'paint the grid cells if they are dirty'
        if self.qdirty:
            n = 20 + self.current_filling/10
            for p in self.qdirty[:n]:
                self.dirty[p] = 1
            self.qdirty = self.qdirty[n:]
        else:
            self.current_filling= 0
            
        while self.dirty:
            ((x,y),dirty) = self.dirty.popitem()
            v = self.grid_get(x,y)

            b = (self.cell_casts_shadow(x, y-1)*2 +
                  self.cell_casts_shadow(x-1, y) * 1 +
                  self.cell_casts_shadow(x-1, y-1) * 4)

            self.surface.blit(self.gridblit[v*8+b], (x*3,y*3))

def blink(t):
    'returns 0 or 1 alternating every t seconds'
    return int(time.time()/t)%2

def main(winstyle = 0):
    # Initialize pygame
    pygame.init()
    load_sounds() # get all wav files from sounds dir

    # Set the display mode
    if FullScreenMode:
        winstyle = FULLSCREEN
#        SCREENRECT     = Rect(0, 0, 1024, 768)
#        grid_size = (333, 205)
        SCREENRECT     = Rect(0, 0, 640, 480)
        grid_size = (194, 122)
        grid_position = (30,85)
    else:
        winstyle = HWPALETTE
        SCREENRECT     = Rect(0, 0, 640, 480)
        grid_size = (194, 122)
        grid_position = (30,85)
        
    pygame.display.set_mode(SCREENRECT.size, winstyle, 16)
    pygame.display.set_caption('Pyx')
    
    # grid and colors
    grid = Grid(grid_size)
    w,h=grid_size

    # background and title
    gsurface = pygame.display.get_surface()
    gsurface.fill(grid.cell_color[3][0])

    # title
    #bigfont = pygame.font.Font(None, 60)
    #title_img = crystal.textCrystal(bigfont, 'Pyx', 10, (64, 128, 255), 70)
    title_img = pygame.image.load('title.png').convert_alpha()
    gsurface.blit(title_img, (273, 10))

    start_image = pygame.image.load('start.png').convert_alpha()
    gameover_image = pygame.image.load('gameover.png').convert_alpha()
    bonus_image = pygame.image.load('bonus.png').convert_alpha()
    level_image = pygame.image.load('level.png').convert_alpha()
    big_player_image = pygame.image.load('big_player.png').convert_alpha()

    # sprites
    grid.sprites = sprites = pygame.sprite.RenderPlain()
    sprites.offset = grid.offset = grid_position
    grid.player = player = Player(sprites, grid, ((w/2)&0xfffe, h-4),)
    grid.pyx = pyx = Pyx(grid)

    # spyrx bar
    spyrx_bar = grid.spyrx_bar = SpyrxBar(grid, SpyrxBarSpeed)

    # score graphics
    digits = load_digits('digits.png', (15,23))
    gsurface.blit(pygame.image.load('percent.png').convert_alpha(), (grid_position[0]+7,50))
    ascii_images = load_ascii('ascii.png', (30,40))
    ascii_gold_images = load_ascii('ascii_gold.png', (30,40))

    # music
    #pygame.mixer.music.load('bach.mid')
    #pygame.mixer.music.play(-1)

    # frame timing
    clock = pygame.time.Clock()
    t0 = time.time()

    # ultra-secure bulletproof encryption for high score list so that even
    # the most sophisticated hacker can't possibly cheat!

    import rotor, traceback
    score_rotor = rotor.newrotor('secret')

    try:
        f = open('scores','r')
        high_scores = eval(score_rotor.decrypt(f.read()))
        f.close()
    except:
        # reset if any error occurs
        traceback.print_exc()
        high_scores = [('',0),('',0),('',0),('',0),('',0)]

    editing_score = -1
    quitting = 0

    while 1:
        # update keyboard state
        pygame.event.pump()

        if pygame.event.get([QUIT]):
            quitting = 1

        if editing_score == -1:
            for e in pygame.event.get([KEYDOWN]):
                if e.key == K_ESCAPE:
                    quitting = 1
                elif e.key == K_SPACE:
                    grid.reset(1)
                    pyx.reset()
                    player.reset()
                    spyrx_bar.reset(1)

        if quitting:
            break

        # update spyrx bar
        spyrx_bar.update()

        #draw the grid        
        grid.paint()
        pygame.display.get_surface().blit(grid.surface, grid_position)

        #draw the sprites
        sprites.update()
        pyx.update()
        dirty = sprites.draw(pygame.display.get_surface())
        #pygame.display.update(dirty)
        pyx.paint(gsurface)

        gx0,gy0 = grid_position
        gx1,gy1 = gx0+w*3, gy0+h*3

        # draw the score
        if player.score_dirty:
            player.score_dirty = 0
            gsurface.fill(grid.cell_color[3][0], (gx0+6,16, 12*15, 23))
            blit_number(gsurface, digits, (15,23), (grid_position[0]+6,16), 1, player.score)
            gsurface.fill(grid.cell_color[3][0], (gx0+6,50, 30, 23))
            blit_number(gsurface, digits, (15,23), (grid_position[0]+6,50), 2, grid.percent,0)

        if grid.level_dirty and player.game_over>-1:
            grid.level_dirty = 0
            gsurface.fill(grid.cell_color[3][0], (gx1-120,16, 120, 23))
            gsurface.blit(level_image, (gx1-68-15*len('%d'%grid.level),16))
            blit_number(gsurface, digits, (15,23), (gx1-2*15-6,16), 2, grid.level )

        if player.lives_dirty:
            player.lives_dirty = 0
            gsurface.fill(grid.cell_color[3][0], (gx0,gy0+h*3, w*3, 24))

            n10 = player.lives/10
            n1 = player.lives%10
            
            for i in range(n10):
                gsurface.blit(big_player_image, (gx0+w*3-6-(i+1)*24, gy0+h*3))

            for i in range(n1):
                gsurface.blit(player.liveimage, (gx0+w*3-6-(i+1)*16-n10*24, gy0+h*3))

        if player.bonus or player.bonus_pause:
            gsurface.blit(bonus_image, (236,247))
            blit_number(gsurface, digits, (15,23), (325,254), 2, player.bonus/1000)
            blit_number(gsurface, digits, (15,23), (325+15*2,254), 3, 000, 0)

            if player.bonus_pause==0:
                if player.bonus > 0:
                    player.bonus -= 1000
                    player.add_score (1000)
                
                player.bonus_pause = 6
            else:
                player.bonus_pause -= 1

        elif player.levelling:
            player.levelling -= 1
            if player.levelling == 0:
                get_sound('level').play()
                grid.level += 1
                grid.level_dirty = 1
                grid.reset()
                pyx.reset()
                spyrx_bar.reset(1, SpyrxBarSpeed+grid.level-1)
                player.position = player.start_position
                player.score_dirty = 1

        if player.game_over:
            if player.game_over == 1:
                # game just ended
                player.game_over = 2
                pygame.event.get([KEYDOWN,KEYUP])

                for i in range(len(high_scores)):
                    (name,score) = high_scores[i]
                    if player.score > score:
                        high_scores.insert(i,(player.name,player.score))
                        high_scores = high_scores[:5]
                        editing_score = i
                        break

            x0,y0=grid_position

            if high_scores:
                blit_text(gsurface, ascii_images, (21,40), (x0+20,y0+20), 'Top 5 scores:')
                y0+=40

                for i in range(len(high_scores)):
                    (name,score) = high_scores[i]
                    if editing_score == i:
                        for e in pygame.event.get([KEYDOWN]):
                            if e.key < 127 and e.key>=32:
                                if e.mod==0:
                                    name += chr(e.key)
                                elif e.mod==2:
                                    if chr(e.key) in shift_map:
                                        name += shift_map[chr(e.key)]
                                name = name[:12]
                            elif e.key == K_BACKSPACE:
                                name = name[:-1]
                            elif e.key == K_RETURN:
                                high_scores[i] = name,score
                                f = open('scores','w')
                                f.write(score_rotor.encrypt(`high_scores`))
                                f.close()
                                editing_score = -1
                            high_scores[i] = name,score

                        if blink(.5):
                            blit_text(gsurface, ascii_gold_images, (21,40), (x0+20,y0+20), '%-12s%10d' % (name+'_',score))
                        else:
                            blit_text(gsurface, ascii_gold_images, (21,40), (x0+20,y0+20), '%-12s%10d' % (name+' ',score))
                    else:
                        blit_text(gsurface, ascii_images, (21,40), (x0+20,y0+20), '%-12s%10d' % (name,score))
                    y0+=35

            if player.game_over==-1:
                gsurface.blit(start_image, (196,360))
            else:
                gsurface.blit(gameover_image, (196,350))

        pygame.display.flip()

        #cap the FrameRate (unless game not in progress)
        if not player.game_over:
            clock.tick(FrameRate+grid.level*2)

        #check the FrameRate
        if ShowFPS and time.time()-t0>1:
            print '%d fps' % int(clock.get_fps())
            t0 = time.time()


if __name__ == '__main__': main()

