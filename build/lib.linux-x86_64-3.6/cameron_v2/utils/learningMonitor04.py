import shutil
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
import matplotlib
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc( 'text', usetex = True )
# matplotlib.rcParams['text.usetex'] = True
class Scalar():
    def __init__( self, idx, name, val = 0 ) :
        self.idx = idx
        self.name = name
        self.val = val

class Plot():
    def __init__( self, idx, name  ):
        self.idx = idx
        self.name = name
        self.x = []
        self.y = []
        self.xrange = None
        self.yrange = None
        self.xlabel = 'xlabel'
        self.ylabel = 'ylabel'
        SAVE_PATH = './Monitor/images/'
        FILE_NAME = 'plot' + str( self.idx ) + '.jpg'
        self.filename_fp = os.path.join( SAVE_PATH, FILE_NAME )

    def generateFigure( self ) :
        self.fig = plt.figure( figsize = ( 8, 5 ) )
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        self.ax = self.fig.add_subplot( 111 )
        plt.title( self.name )
        if ( self.x is not None )and( self.y is not None ) :
            plt.xlabel( self.xlabel )
            plt.ylabel( self.ylabel )

            self.ax.plot( self.x, self.y )
            if self.yrange is not None :
                plt.ylim( self.yrange )
            self.fig.savefig( self.filename_fp )

class Weight():
    def __init__( self, idx, name ) :
        self.idx = idx
        self.name = name
        self.weight = None
        SAVE_PATH = './Monitor/images/'
        FILE_NAME = 'weight' + str( self.idx ) + '.jpg'
        self.filename_fp = os.path.join( SAVE_PATH, FILE_NAME )

    def generateFigure( self ) :
        self.fig = plt.figure( figsize = ( 5, 5 ) )
        self.ax = self.fig.add_subplot( 111 )
        plt.title( self.name )
        if self.weight is not None :
            print(self.weight.shape)
            im = self.ax.imshow( self.weight, animated = False, cmap = plt.cm.jet )
            self.fig.colorbar( im )
            self.fig.savefig( self.filename_fp )

class Hist():
    def __init__( self, idx, name ) :
        self.idx = idx
        self.name = name
        self.xlabel = 'xlabel'
        self.ylabel = 'ylabel'
        self.data = None
        self.bins = None
        self.rwidth = 0.8
        SAVE_PATH = './Monitor/images/'
        FILE_NAME = 'hist' + str( self.idx ) + '.jpg'
        self.filename_fp = os.path.join( SAVE_PATH, FILE_NAME )

    def generateFigure( self ) :
        self.fig = plt.figure( figsize = ( 8, 5 ) )
        self.ax = self.fig.add_subplot( 111 )
        plt.title( self.name )
        if self.data is not None :
            plt.hist( self.data, self.bins, histtype = 'bar', rwidth = self.rwidth)
            self.fig.savefig( self.filename_fp )

class ScatterData():
    def __init__( self, x, y, c, legend ) :
        self.x = x
        self.y = y
        self.c = c
        self.legend = legend

class CmapScatter():
    def __init__( self, idx, name ) :
        self.idx = idx
        self.name = name
        self.xlabel = 'xlabel'
        self.ylabel = 'ylabel'
        self.scatter_data = None
        SAVE_PATH = './Monitor/images/'
        FILE_NAME = 'cmapscatter' + str( self.idx ) + '.jpg'
        self.filename_fp = os.path.join( SAVE_PATH, FILE_NAME )

    def generateFigure( self ) :
        self.fig = plt.figure( figsize = ( 8, 5 ) )
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        self.ax = self.fig.add_subplot( 111 )
        plt.title( self.name )
        cm = plt.cm.get_cmap( 'RdYlBu' )
        sc = plt.scatter( x = self.scatter_data.x, y = self.scatter_data.y,
                          c = self.scatter_data.c,
                          label = self.scatter_data.legend, cmap = cm  )
        plt.colorbar( sc )
        plt.legend( loc = 'best' )
        self.fig.savefig( self.filename_fp )



class LearningMonitor() :
    def __init__( self, update_cycle = 1. ) :
        # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        # rc( 'text', usetex = True )
        self.MONITOR_DIR = './Monitor/'
        self.SCRIPT_FILE_PATH = os.path.join( self.MONITOR_DIR, 'script.js' )
        self.IMG_SAVE_FOLDER = os.path.join( self.MONITOR_DIR, 'images' )
        self.MAX_SCALARS = 10
        self.MAX_PLOTS = 10
        self.MAX_WEIGHTS = 10
        self.MAX_HISTS = 6
        self.MAX_CMAPSCATTERS = 6
        self.NUM_SCALARS = 0
        self.NUM_PLOTS = 0
        self.NUM_WEIGHTS = 0
        self.NUM_HISTS = 0
        self.NUM_CMAPSCATTERS = 0
        self.UPDATE_COUNTER = 0
        self.cmapscatters = []
        self.plots = []
        self.weights = []
        self.hists = []
        self.scalars = []
        self._deleteMonitorDir()
        self._setupMonitorDir()
        self._deleteAllFigures()

    def _deleteAllFigures( self ) :
        for filename in os.listdir( self.IMG_SAVE_FOLDER ):
            if filename.endswith(".jpg"):
                os.remove( os.path.join( self.IMG_SAVE_FOLDER, filename ) )


    def setCmapScatter( self, name, x, y, c, legend = None, xlabel = None, ylabel = None ) :
        ## CHECK IF THE NAME IS RESERVED ##################
        all_names = []
        all_idxs = []
        for target in self.cmapscatters :
            all_names.append( target.name )
            all_idxs.append( target.idx )
        if name not in all_names :
            if len( self.cmapscatters ) == self.MAX_CMAPSCATTERS :
                sys.exit( 'No more Scatter can be added' )
            if len( all_idxs ) > 0 :
                new_idx = int( np.max( all_idxs ) ) + 1
            else :
                new_idx = 0
            self.cmapscatters.append( CmapScatter( idx = new_idx, name = name ))
            self.NUM_CMAPSCATTERS += 1
        ###################################################

        idx = None
        for target in self.cmapscatters :
            if target.name == name :
                idx = target.idx
        if idx is None :
            sys.exit( 'No Scatter named ' + name )
        self.cmapscatters[ idx ].xlabel = xlabel
        self.cmapscatters[ idx ].ylabel = ylabel
        self.cmapscatters[ idx ].scatter_data = ScatterData( x, y, c, legend )

    def setPlot( self, name, x, y, xlabel = None, ylabel = None, xlim = None, ylim = None ) :
        ## CHECK IF THE NAME IS RESERVED ##################
        all_names = []
        all_idxs = []
        for target in self.plots :
            all_names.append( target.name )
            all_idxs.append( target.idx )
        if name not in all_names :
            if len( self.plots ) == self.MAX_PLOTS :
                sys.exit( 'No more Plot object can be added' )
            if len( all_idxs ) > 0 :
                new_idx = int( np.max( all_idxs ) ) + 1
            else :
                new_idx = 0
            self.plots.append( Plot( idx = new_idx, name = name ))
            self.NUM_PLOTS += 1
        ###################################################

        idx = None
        for target in self.plots :
            if target.name == name :
                idx = target.idx
        if idx is None :
            sys.exit( 'No Plot named ' + name )
        self.plots[ idx ].x = x
        self.plots[ idx ].y = y
        self.plots[ idx ].xlabel = xlabel
        self.plots[ idx ].ylabel = ylabel
        self.plots[ idx ].xlim = xlim
        self.plots[ idx ].ylim = ylim

    def setPlotByAppend( self, name, x, y, xlabel = None , ylabel = None, xlim = None, ylim = None ) :
        ## CHECK IF THE NAME IS RESERVED ##################
        all_names = []
        all_idxs = []
        for target in self.plots :
            all_names.append( target.name )
            all_idxs.append( target.idx )
        if name not in all_names :
            if len( self.plots ) == self.MAX_PLOTS :
                sys.exit( 'No more Plot object can be added.' )
            if len( all_idxs ) > 0 :
                new_idx = int( np.max( all_idxs ) ) + 1
            else :
                new_idx = 0
            self.plots.append( Plot( idx = new_idx, name = name ))
            self.NUM_PLOTS += 1
        ###################################################

        idx = None
        for target in self.plots :
            if target.name == name :
                idx = target.idx
        if idx is None :
            sys.exit( 'No Plot named ' + name )

        self.plots[ idx ].x.append( x )
        self.plots[ idx ].y.append( y )
        self.plots[ idx ].xlabel = xlabel
        self.plots[ idx ].ylabel = ylabel
        self.plots[ idx ].xlim = xlim
        self.plots[ idx ].ylim = ylim


    def setWeight( self, name, weight ) :
        ## CHECK IF THE NAME IS RESERVED ##################
        all_names = []
        all_idxs = []
        for target in self.weights :
            all_names.append( target.name )
            all_idxs.append( target.idx )
        if name not in all_names :
            if len( self.weights) == self.MAX_WEIGHTS :
                sys.exit( 'No more Weight can be added' )
            if len( all_idxs ) > 0 :
                new_idx = int( np.max( all_idxs ) ) + 1
            else :
                new_idx = 0
            self.weights.append( Weight( idx = new_idx, name = name ))
            self.NUM_WEIGHTS += 1
        ###################################################

        idx = None
        for target in self.weights :
            if target.name == name :
                idx = target.idx
        if idx is None :
            sys.exit( 'No Weight named ' + name )
        self.weights[ idx ].weight = weight


    def setHist( self, name, data, xlabel = None, ylabel = None, bins = None, rwidth = 0.8, normed=False ) :
        ## CHECK IF THE NAME IS RESERVED ##################
        all_names = []
        all_idxs = []
        for target in self.hists :
            all_names.append( target.name )
            all_idxs.append( target.idx )
        if name not in all_names :
            if len( self.hists ) == self.MAX_HISTS :
                sys.exit( 'No more Hist object can be added.' )
            if len( all_idxs ) > 0 :
                new_idx = int( np.max( all_idxs ) ) + 1
            else :
                new_idx = 0
            self.hists.append( Hist( idx = new_idx, name = name ))
            self.NUM_HISTS += 1
        ###################################################

        idx = None
        for target in self.hists :
            if target.name == name :
                idx = target.idx
        if idx is None :
            sys.exit( 'No Hist named ' + name )
        self.hists[ idx ].data = data
        self.hists[ idx ].xlabel = xlabel
        self.hists[ idx ].ylabel = ylabel
        self.hists[ idx ].bins = bins
        self.hists[ idx ].rwidth = rwidth

    def setScalar( self, name, val ) :
        ## CHECK IF THE NAME IS RESERVED ##################
        all_names = []
        all_idxs = []
        for target in self.scalars :
            all_names.append( target.name )
            all_idxs.append( target.idx )
        if name not in all_names :
            if len( self.scalars ) == self.MAX_SCALARS :
                sys.exit( 'No more Scalar object can be added' )
            if len( all_idxs ) > 0 :
                new_idx = int( np.max( all_idxs ) ) + 1
            else :
                new_idx = 0
            self.scalars.append( Scalar( idx = new_idx, name = name ))
            self.NUM_SCALARS += 1
        ###################################################

        idx = None
        for target in self.scalars :
            if target.name == name :
                idx = target.idx
        if idx is None :
            sys.exit( 'No Scalar named ' + name )
        self.scalars[ idx ].val = val


    def setValFromLabel( self, label, val ) :
        for i in range( self.MAX_SCALARS ) :
            if self.scalars[ i ].label == label :
                self.scalars[ i ].val = val

    def printScalars( self ) :
        txt = "-\n"
        for i in range( len( self.scalars ) ) :
            txt = txt + self.scalars[i].label + ":" + str( self.scalars[i].val ) + ' | '
        print(txt)

    def update( self ) :
        self.updateScalars()
        for i in range( self.NUM_PLOTS ) :
            self.plots[ i ].generateFigure()
        for i in range( self.NUM_WEIGHTS ) :
            self.weights[ i ].generateFigure()
        for i in range( self.NUM_HISTS ) :
            self.hists[ i ].generateFigure()
        for i in range( self.NUM_CMAPSCATTERS ) :
            self.cmapscatters[ i ].generateFigure()
        for i in range( 20 ) : plt.close()

    def sparseUpdate( self, update_cycle ) :
        self.UPDATE_COUNTER += 1
        if self.UPDATE_COUNTER % update_cycle == 0 :
            self.update()

    def updateScalars( self ) :
        with open( self.SCRIPT_FILE_PATH, 'r') as myfile:
            data = myfile.read()

        pattern = re.compile( r'(var\s+)(num_scalars\s*=\s*)(\d+)(;)' )
        matches = pattern.finditer( data )
        for match in matches :
            new = 'var num_scalars = ' + str( self.MAX_SCALARS ) + ';'
            data = data.replace( match.group( 0 ), new )

        pattern = re.compile( r'(var\s*)(scalar)(\d)(_val\s*=\s*)([+-]?\d+[.]?\d*;)' )
        matches = pattern.finditer( data )
        for match in matches :
            idx =  int( match.group( 3 ) )
            if idx < self.NUM_SCALARS :
                new = 'var scalar' + str( idx ) + '_val = ' + str( self.scalars[ idx ].val ) + ';'
                data = data.replace( match.group( 0 ), new )

        pattern = re.compile( r'(var\s*)(scalar)(\d)(_label\s*=\s*)("\w+";)' )
        matches = pattern.finditer( data )
        for match in matches :
            idx =  int( match.group( 3 ) )
            if idx < self.NUM_SCALARS :
                new = 'var scalar' + str( idx ) + '_label = "' +  self.scalars[ idx ].name + '";'
                data = data.replace( match.group( 0 ), new )

        with open( self.SCRIPT_FILE_PATH, "w") as text_file:
            text_file.write( data )

    def _setupMonitorDir( self ) :
        if not os.path.exists( self.MONITOR_DIR ):
            os.makedirs( self.MONITOR_DIR )
            os.makedirs( os.path.join( self.MONITOR_DIR, 'images' ))

            with open( os.path.join( self.MONITOR_DIR, 'index.html' ), 'w+' ) as index_file :
                index_file.write( index_txt )
            with open( os.path.join( self.MONITOR_DIR, 'style.css' ), 'w+' ) as style_file :
                style_file.write( style_txt )
            with open( os.path.join( self.MONITOR_DIR, 'script.js' ), 'w+' ) as script_file :
                script_file.write( script_txt )

        print('---')
        print('Open this in the browser:')
        print('\'' + 'file://' + os.path.abspath( os.path.join( self.MONITOR_DIR, 'index.html' ) ) + '\'')
        print('---')
    def _deleteMonitorDir( self ) :
        if os.path.exists( self.MONITOR_DIR ) :
            shutil.rmtree( self.MONITOR_DIR )

index_txt = '''
<!DOCTYPE html>
<html class = "no-js">

  <head>
    <meta charset = "utf-8">
    <title>Learning Monitor 4</title>
    <meta name = "description" content = "">
    <meta name = "viewpoint" content = "width=device-width">
    <link rel = "stylesheet" href = "./style.css">
  </head>

  <body>
    <!-- <h1>Learning Monitor</h1> -->
    <div class = "wrapper">
      <ul>
      <li class = "scalar" id = "scalar0"></li>
      <li class = "scalar" id = "scalar1"></li>
      <li class = "scalar" id = "scalar2"></li>
      <li class = "scalar" id = "scalar3"></li>
      <li class = "scalar" id = "scalar4"></li>
      <li class = "scalar" id = "scalar5"></li>
      <li class = "scalar" id = "scalar6"></li>
      <li class = "scalar" id = "scalar7"></li>
      <li class = "scalar" id = "scalar8"></li>
      <li class = "scalar" id = "scalar9"></li>
    </ul>


      <img class = "plot" src = "./images/plot0.jpg" onerror="this.style.display='none'">
      <img class = "plot" src = "./images/plot1.jpg" onerror="this.style.display='none'">
      <img class = "plot" src = "./images/plot2.jpg" onerror="this.style.display='none'">
      <img class = "plot" src = "./images/plot3.jpg" onerror="this.style.display='none'">
      <img class = "plot" src = "./images/plot4.jpg" onerror="this.style.display='none'">
      <img class = "plot" src = "./images/plot5.jpg" onerror="this.style.display='none'">
      <img class = "plot" src = "./images/plot6.jpg" onerror="this.style.display='none'">
      <img class = "plot" src = "./images/plot7.jpg" onerror="this.style.display='none'">
      <img class = "plot" src = "./images/plot8.jpg" onerror="this.style.display='none'">
      <img class = "plot" src = "./images/plot9.jpg" onerror="this.style.display='none'">


      <img class = "hist" src = "./images/hist0.jpg" onerror="this.style.display='none'">
      <img class = "hist" src = "./images/hist1.jpg" onerror="this.style.display='none'">
      <img class = "hist" src = "./images/hist2.jpg" onerror="this.style.display='none'">
      <img class = "hist" src = "./images/hist3.jpg" onerror="this.style.display='none'">
      <img class = "hist" src = "./images/hist4.jpg" onerror="this.style.display='none'">
      <img class = "hist" src = "./images/hist5.jpg" onerror="this.style.display='none'">

      <img class = "weight" src = "./images/weight0.jpg" onerror="this.style.display='none'">
      <img class = "weight" src = "./images/weight1.jpg" onerror="this.style.display='none'">
      <img class = "weight" src = "./images/weight2.jpg" onerror="this.style.display='none'">
      <img class = "weight" src = "./images/weight3.jpg" onerror="this.style.display='none'">
      <img class = "weight" src = "./images/weight4.jpg" onerror="this.style.display='none'">
      <img class = "weight" src = "./images/weight5.jpg" onerror="this.style.display='none'">
      <img class = "weight" src = "./images/weight6.jpg" onerror="this.style.display='none'">
      <img class = "weight" src = "./images/weight7.jpg" onerror="this.style.display='none'">
      <img class = "weight" src = "./images/weight8.jpg" onerror="this.style.display='none'">
      <img class = "weight" src = "./images/weight9.jpg" onerror="this.style.display='none'">

      <img class = "cmapscatter" src = "./images/cmapscatter0.jpg" onerror="this.style.display='none'">
      <img class = "cmapscatter" src = "./images/cmapscatter1.jpg" onerror="this.style.display='none'">
      <img class = "cmapscatter" src = "./images/cmapscatter2.jpg" onerror="this.style.display='none'">
      <img class = "cmapscatter" src = "./images/cmapscatter3.jpg" onerror="this.style.display='none'">
      <img class = "cmapscatter" src = "./images/cmapscatter4.jpg" onerror="this.style.display='none'">
      <img class = "cmapscatter" src = "./images/cmapscatter5.jpg" onerror="this.style.display='none'">

      <script src = "./script.js"></script>
    </div>
  </body>
  <main>
  </main>
</html>
'''

script_txt = '''
var num_scalars = 10;

var scalar0_label = "scalar0";
var scalar1_label = "scalar1";
var scalar2_label = "scalar2";
var scalar3_label = "scalar3";
var scalar4_label = "scalar4";
var scalar5_label = "scalar5";
var scalar6_label = "scalar6";
var scalar7_label = "scalar7";
var scalar8_label = "scalar8";
var scalar9_label = "scalar9";

var scalar0_val = 0;
var scalar1_val = 0;
var scalar2_val = 0;
var scalar3_val = 0;
var scalar4_val = 0;
var scalar5_val = 0;
var scalar6_val = 0;
var scalar7_val = 0;
var scalar8_val = 0;
var scalar9_val = 0;

document.getElementById( "scalar0" ).innerHTML = scalar0_label + " : " + scalar0_val;
document.getElementById( "scalar1" ).innerHTML = scalar1_label + " : " + scalar1_val;
document.getElementById( "scalar2" ).innerHTML = scalar2_label + " : " + scalar2_val;
document.getElementById( "scalar3" ).innerHTML = scalar3_label + " : " + scalar3_val;
document.getElementById( "scalar4" ).innerHTML = scalar4_label + " : " + scalar4_val;
document.getElementById( "scalar5" ).innerHTML = scalar5_label + " : " + scalar5_val;
document.getElementById( "scalar6" ).innerHTML = scalar6_label + " : " + scalar6_val;
document.getElementById( "scalar7" ).innerHTML = scalar7_label + " : " + scalar7_val;
document.getElementById( "scalar8" ).innerHTML = scalar8_label + " : " + scalar8_val;
document.getElementById( "scalar9" ).innerHTML = scalar9_label + " : " + scalar9_val;


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var i = 0;

function update()
{

  location.reload();
}

var nIntervId;

function updateEverySeconds() {
  nIntervId = setInterval( update, 4000);
}

updateEverySeconds();
'''

style_txt = '''
h1
{
  font-size: 30px;
  font-family: Sans-serif	;
}

li
{
  background-color : #E8E8E8;
  font-size: 25px;
  margin: 0.5%;
  padding: 0.7%;
  font-family: Sans-serif	;
  display: inline;

}
.scalar
{
}

.plot
{
  width: 600px;

}

.hist
{
  width: 600px;
}

.weight
{
  width: 400px;
}

.cmapscatter
{
  width: 600px;
 }
'''
