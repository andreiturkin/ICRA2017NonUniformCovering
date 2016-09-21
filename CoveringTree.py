from ete3 import Tree
from math import sqrt
import numpy as np

#Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#OS
import os

# Date and Time
import datetime

#Images
from PIL import Image

class Rect:
    def __init__(self, left, top, width, height):
        self.left = left
        self.right = left + width
        self.top = top
        self.bottom = top + height
        self.width = width
        self.height = height
        self.centerx = left + width/2
        self.centery = top + height/2
        self.center = (left + width/2, top + height/2)
    
    def __str__(self):
        return '<Rect: {}, {}, {}, {}>'.format(self.left, self.top, self.width, self.height)  
        
class CoveringTree:
############################################################################################
# Constructor
############################################################################################    

    def __init__(self, l0, l1_bounds, l2_bounds, idelta=0):
        #Initialize Base
        self.__l0 = l0
        
        #Initialize the Bounds
        self.__l1_bounds = l1_bounds
        self.__l2_bounds = l2_bounds
        
        #Define Initial Rectangle P
        left = -self.__l1_bounds[1]
        top = 0
        width = self.__l1_bounds[1]+self.__l0+self.__l2_bounds[1]
        height = min(self.__l1_bounds[1],self.__l2_bounds[1])
        
        #Initialize initial Space where the workspace lie
        self.__Xspace = Rect(left, top, width, height)
        #Initialize the Root
        self.__initTree(self.__Xspace)
                
        #Initialize the minimal size of the rectangle
        self.__delta = idelta
        
        #Initialize plotting facilities
        self.__fig = plt.figure()
               
        self.__ax = self.__fig.add_subplot(111)
        self.__ax.axis('scaled')
        #self.__ax.axis([self.__Xspace.left-1, self.__Xspace.right+1, self.__Xspace.top-1, self.__Xspace.bottom+1 ])
        self.__ax.axis([self.__Xspace.left, self.__Xspace.right, self.__Xspace.top, self.__Xspace.bottom ])
  
############################################################################################
# Private Members
############################################################################################        
    def __vSplitter(self, iRect):
        newleft1 = iRect.left
        newtop1 = iRect.top
        newwidth1 = iRect.width/2.0
        newheight1 = iRect.height
        Rleft = Rect(newleft1, newtop1, newwidth1, newheight1)
        
        newleft2 = iRect.left + iRect.width/2.0
        newtop2 = iRect.top
        newwidth2 = iRect.width/2.0
        newheight2 = iRect.height
        Rright = Rect(newleft2, newtop2, newwidth2, newheight2)
        return Rleft, Rright
    
    def __hSplitter(self, iRect):
        newleft1 = iRect.left
        newtop1 = iRect.top
        newwidth1 = iRect.width
        newheight1 = iRect.height/2.0
        Rleft = Rect(newleft1, newtop1, newwidth1, newheight1)
        
        newleft2 = iRect.left
        newtop2 = iRect.top + iRect.height/2.0
        newwidth2 = iRect.width
        newheight2 = iRect.height/2.0
        Rright = Rect(newleft2, newtop2, newwidth2, newheight2)
        return Rleft, Rright
    
    def __d(self, iRect):
        return sqrt(iRect.width**2.0 + iRect.height**2.0)
    
    def __PyCppAD_g_for_Optimize(self, x, arg):
        return arg.forward(0, x) 

    def __PyCppAD_dg_for_Optimize(self, x, arg):
        return arg.jacobian(x).flatten()
    
    def __g1(self, x):
        return np.array([x[0]**2.0 + x[1]**2.0 - (self.__l1_bounds[1]**2.0)]) 
    
    def __g2(self, x):
        return np.array([self.__l1_bounds[0]**2.0 - (x[0]**2.0) - (x[1]**2.0)])
    
    def __g3(self, x):
        return np.array([(x[0]**2.0) + (x[1]**2.0) - (self.__l2_bounds[1]**2.0)])
        
    def __g4(self, x):
        return np.array([self.__l2_bounds[0]**2.0 - (x[0]**2.0) - (x[1]**2.0)])
    
    def __analyseRect(self, iRect):
        g_min = []
        g_max = []
        
        xmin = iRect.left
        xmax = iRect.left + iRect.width
        ymin = iRect.top
        ymax = iRect.top + iRect.height
        
        #MIN
        #g1(x1,x2)
        a1min = min(abs(xmin),abs(xmax))
        a2min = min(abs(ymin),abs(ymax))
        g_min.append(self.__g1((a1min,a2min)))
        #g2(x1,x2)
        a1min = max(abs(xmin),abs(xmax))
        a2min = max(abs(ymin),abs(ymax))
        g_min.append(self.__g2((a1min,a2min)))
        #g3(x1,x2)
        a1min = min(abs(xmin-self.__l0),abs(xmax-self.__l0))
        a2min = min(abs(ymin),abs(ymax))
        g_min.append(self.__g3((a1min,a2min)))
        #g4(x1,x2)
        a1min = max(abs(xmin-self.__l0),abs(xmax-self.__l0))
        a2min = max(abs(ymin),abs(ymax))
        g_min.append(self.__g4((a1min,a2min)))
        
        #MAX
        #g1(x1,x2)
        a1max = max(abs(xmin),abs(xmax))
        a2max = max(abs(ymin),abs(ymax))
        g_max.append(self.__g1((a1max,a2max)))
        #g2(x1,x2)
        a1max = min(abs(xmin),abs(xmax))
        a2max = min(abs(ymin),abs(ymax))
        g_max.append(self.__g2((a1max,a2max)))
        #g3(x1,x2)
        a1max = max(abs(xmin-self.__l0),abs(xmax-self.__l0))
        a2max = max(abs(ymin),abs(ymax))
        g_max.append(self.__g3((a1max,a2max)))
        #g4(x1,x2)
        a1max = min(abs(xmin-self.__l0),abs(xmax-self.__l0))
        a2max = min(abs(ymin),abs(ymax))
        g_max.append(self.__g4((a1max,a2max))) 
        
        #There is no solution for the rectangle
        for g_mini in g_min:
            if (g_mini>0):
                #mark it as out of range
                inrange = False
                return False, inrange
            
        #The rectangle is a part of the solution
        leq = True
        for g_maxi in g_max:
            leq = leq and (g_maxi<=0)
        
        if leq:
            #mark it as in range
            inrange = True
            return False, inrange 
        
        #The rectangle has to be processed further
        return True, False

    def __addToTree(self, motherNode, iRect1, iRect2, childNodeLevel):
        # and add the nodes as children.
        oNode2 = motherNode.add_child(name='{}'.format(childNodeLevel))
        oNode1 = motherNode.add_child(name='{}'.format(childNodeLevel))
        #add features
        oNode2.add_feature('Rect',iRect2)
        oNode1.add_feature('Rect',iRect1)
        
    def __getNewRect(self, iRect, level):        
        (oRleft,oRright) = self.__vSplitter(iRect) if (level%2==0) else self.__hSplitter(iRect)
        return (oRleft,oRright)
    
    def __initTree(self, Xspace):
        self.__sTree = Tree('0;') #name here is the level of the tree
        motherNode = self.__sTree.search_nodes(name='0')[0]
        motherNode.add_feature('Rect',Xspace)
    
    def __drawRect(self, iRect, fillIt, PlotEdges=True, inQI=False, inQE=True):
        if(PlotEdges):
            #Internal
            if inQI and inQE:
                edgeColor = 'black'
                LineStyle='solid'
                LineWidth = 1
                Alpha=0.3
            #External
            if inQE and (not inQI):
                edgeColor = 'red'
                LineStyle='solid'
                LineWidth = 1
                Alpha=None
            #Out of range
            if (not inQE) and (not inQI):
                edgeColor = 'green'
                LineStyle='solid'
                LineWidth = 1
                Alpha=None
            
            self.__ax.add_patch(
                          patches.Rectangle(
                                            (iRect.left, iRect.top),   # (x,y)
                                            iRect.width,          # width    
                                            iRect.height,         # height
                                            fill = inQI,
                                            alpha = Alpha,
                                            linestyle = LineStyle,
                                            edgecolor = edgeColor,  
                                            lw = LineWidth)
                          )
        else:
            self.__ax.add_patch(
                          patches.Rectangle(
                                            (iRect.left, iRect.top),   # (x,y)
                                            iRect.width,          # width    
                                            iRect.height,         # height
                                            fill = fillIt,
                                            edgecolor = 'none')
                          )
        plt.draw()
    
    def __AddMarks(self, curLevel, diam):
        self.__tleveltext.set_text('Tree Level = {}'.format(curLevel))
        self.__curdiam.set_text('d = {}'.format(round(diam,4)))
        plt.draw()
############################################################################################
# Public Members
############################################################################################    
    def getCovering(self, maxLevels):
        
        cdRect = self.__d(self.__Xspace)
        print 'The diameter of the initial rectangle is {}\n'.format(cdRect)
        
        bExit = False
        for curLevel in range(0, maxLevels):
            print 'Processing level {}'.format(curLevel)
            
            #Get all the rectangles that are on some level of the tree
            curLevelNodes = self.__sTree.get_leaves_by_name(name='{}'.format(curLevel))
            #Loop over the rectangles
            for curLevelNode in curLevelNodes:
                #Get a rectangle from the tree level
                oRect = curLevelNode.Rect
                #Save current rectangle diameter
                if self.__d(oRect) < cdRect:
                    cdRect = self.__d(oRect)
                    print 'Current level diameter of the rectangle is {}\n'.format(cdRect)
                
                inQE = False
                inQI = False
                #The diameter of the rectangle is less than or equal to the predefined delta value
                if self.__d(oRect) <= self.__delta:
                    #It is too small to decide upon -> save it as if it was in range
                    cont = False
                    inrange = True
                    inQE = True
                    inQI = False
                    #Return the result on the next iteration 
                    bExit = True
                #Otherwise
                else:
                    #Analyze it
                    (cont, inrange) = self.__analyseRect(oRect)
                    if inrange: 
                        inQI = True
                        inQE = True
                #Save the obtained results
                if cont and (curLevel < maxLevels-1):  
                    (oRleft,oRright) = self.__getNewRect(oRect,curLevel)
                    self.__addToTree(curLevelNode, oRleft, oRright, curLevel + 1)
                else:
                    #save results to the analyzed node
                    curLevelNode.add_feature('Inrange',inrange)
                    curLevelNode.add_feature('inQI',inQI)
                    curLevelNode.add_feature('inQE',inQE)
                    
            #All of the rectangles could be obtained on the next iterations are too small
            #so break it
            if bExit:
                print 'The result is obtained for {} levels'.format(curLevel)
                break
    
    def getRectsDist(self, R1, R2):
        d = 1000#float("inf")
        attr = [('left','bottom'), ('right','bottom'), ('left','top'), ('right','top')]
        for i in attr:
            for j in attr:
                icornerx = getattr(R1, i[0])
                icornery = getattr(R1, i[1])
                jcornerx = getattr(R2, j[0])
                jcornery = getattr(R2, j[1])
                cd = sqrt((icornerx-jcornerx)**2 + (icornery-jcornery)**2)
                if cd < d:
                    d = cd
                    c = i 
        return d, c
    
    def getHausdorffDistance(self):
        QI = []
        QJ = []
        for leaf in self.__sTree.iter_leaves():
            if leaf.inQI:
                cr = leaf.Rect
                corners = [('left','bottom'), ('right','bottom'), ('left','top'), ('right','top')]
                for p in corners:
                    QI.append((getattr(cr, p[0]),getattr(cr,p[1])));
            if (leaf.inQE) and (not leaf.inQI):
                cr = leaf.Rect
                corners = [('left','bottom'), ('right','bottom'), ('left','top'), ('right','top')]
                for p in corners:
                    QJ.append((getattr(cr, p[0]),getattr(cr,p[1])));
                    
        h_QItoQJ = 0
        QI_maxd_point = (0,0)
         
        h_QJtoQI = 0
        for p_inQJ in QJ:
            mindist = 1000
            for p_inQI in QI:
                dist = sqrt((p_inQI[0]-p_inQJ[0])**2 + (p_inQI[1]-p_inQJ[1])**2)
                if dist < mindist:
                    mindist = dist
                    corners = p_inQJ
            if mindist > h_QJtoQI:
                h_QJtoQI = mindist
                QJ_maxd_point = p_inQJ
         
        props = [h_QItoQJ, h_QJtoQI, QI_maxd_point, QJ_maxd_point]
        return max(h_QItoQJ, h_QJtoQI), props 
    
    def saveDistance(self, props, fileName='./Images/{0}__{1:02d}_{2:02d}_{3:02d}_covering.jpeg'.format(datetime.date.today(), \
                                                           datetime.datetime.now().hour,\
                                                           datetime.datetime.now().minute,\
                                                           datetime.datetime.now().second),\
                                                           ZoomIn = False, Grayscale = False, ResOnly = False):
        plt.cla()
        if ResOnly:
            for leaf in self.__sTree.iter_leaves():
                #Draw the rectangle without edges
                self.__drawRect(leaf.Rect, leaf.Inrange, False, leaf.inQI, leaf.inQE)
            
            self.__ax.add_patch(patches.Circle((0,0), self.__l1_bounds[0], fill = False, lw = 1, ls='dashed', color='black'))
            self.__ax.add_patch(patches.Circle((0,0), self.__l1_bounds[1], fill = False, lw = 1, ls='dashed', color='black'))
            self.__ax.add_patch(patches.Circle((self.__l0,0), self.__l2_bounds[0], fill = False, lw = 1, ls='dashed', color='black'))
            self.__ax.add_patch(patches.Circle((self.__l0,0), self.__l2_bounds[1], fill = False, lw = 1, ls='dashed', color='black'))
        else:
            for leaf in self.__sTree.iter_leaves():
                #Draw the rectangle with edges
                self.__drawRect(leaf.Rect, leaf.Inrange, True, leaf.inQI, leaf.inQE)
        
        plt.draw()
        if (Grayscale):
            self.__fig.savefig('./Images/temp.png', dpi = 1200)
            Image.open('./Images/temp.png').convert("L").save(fileName)
        else:
            self.__fig.savefig(fileName, dpi = 1200)
        
        if ResOnly:
            return
        
        h_QItoQJ = props[0] 
        h_QJtoQI = props[1]
        QI_maxd_point = props[2]
        QJ_maxd_point = props[3]
        
        if (h_QItoQJ == max(h_QItoQJ, h_QJtoQI)):
            self.__ax.add_patch(patches.Circle(QI_maxd_point, h_QItoQJ, fill = True, alpha=0.1, lw = 1, color='red'))
            self.__ax.scatter(QI_maxd_point[0], QI_maxd_point[1], s=4, c='black')
        else:
            self.__ax.add_patch(patches.Circle(QJ_maxd_point, h_QJtoQI, fill = True, alpha=0.1, lw = 1, color='red'))
            self.__ax.scatter(QJ_maxd_point[0], QJ_maxd_point[1], s=4, c='black')        
        
        plt.draw()
        if (ZoomIn):
            name, extension = os.path.splitext(fileName)
            name += '_dist_zoomed'
            fileName = name + extension
            
            if (h_QItoQJ == max(h_QItoQJ, h_QJtoQI)):
                self.__ax.axis([QI_maxd_point[0]-h_QItoQJ, QI_maxd_point[0]+h_QItoQJ, 
                                QI_maxd_point[1]-h_QItoQJ, QI_maxd_point[1]+h_QItoQJ])
            else:
                self.__ax.axis([QJ_maxd_point[0]-h_QJtoQI, QJ_maxd_point[0]+h_QJtoQI, 
                                QJ_maxd_point[1]-h_QJtoQI, QJ_maxd_point[1]+h_QJtoQI])
            
            plt.draw()
            
            if (Grayscale):
                self.__fig.savefig('./Images/temp.png', dpi = 1200)
                Image.open('./Images/temp.png').convert("L").save(fileName)
            else:
                self.__fig.savefig(fileName, dpi = 1200)
                