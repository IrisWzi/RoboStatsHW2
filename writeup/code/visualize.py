#! /usr/bin/env python
import sys
import vtk
from numpy import random

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        # Setup the colors array
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName("colors")

    def addPoint(self, point, color):
        # Add the self.colors we created to the self.colors array
        self.colors.InsertNextTupleValue(color)
        # print self.colors
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
             
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        
    def readFile(self, filename):
        self.points = []
        r = [255, 0, 0]
        g = [0, 255, 0]
        b = [0, 0, 255]
        y = [0, 255, 255]
        w = [255, 255, 255]
        p = [100, 100, 100]

        colorCode = dict({1100: w, 1004: g, 1103: r, 1200: p, 1400: b})

        with open(filename, 'r') as fh:
            for i, line in enumerate(fh.readlines()):
                item = line.rstrip() # strip off newline and any other trailing whitespace
                if len(item) == 0 or item[0] == '#':
                    # Comment or blank item
                    continue
                lvals = item.split()
                pos = [float(x) for x in lvals[0:3]]
                lidx = 4 if len(lvals) > 4 else 3

                color = int(lvals[lidx])
                # self.points.append(pos)
                # print colorCode[color]
                self.addPoint(pos, colorCode[color])
        print(self.colors)
        self.vtkPolyData.GetPointData().SetScalars(self.colors)
