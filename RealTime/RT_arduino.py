
"""

LIST OF CLASSES/FUNCTIONS

Arduino (Class)

"""


import os
import sys
import time
import serial
import threading
import queue
import datetime
import time
import warnings
from threading import Thread



class Arduino():
    
    
    """ 
    -------------------------------------------------------------------------------------
    
    Base class for working with Arduino in order to transmit and receive digital signals.

    -------------------------------------------------------------------------------------
    
    Methods:
    
        - init
        - initialize
        - stop
        - io_config
        - io_transmitter
        - io_send
        - digitalHigh
        - digitalLow
        - handshake
        - flushInput
    
    Attributes (see __init__ for details):
    
        - ser
        - port
        - keys_dout
        - keys_din
        - input_sts
        - cmnds
        - cmndflg
        - q_tasks
        - q_inputs
        - state
 
    """
    
    def __init__(self, port, keys_dout=None, keys_din=None, baudrate=115200, timeout=1):
        
        """ 
        -------------------------------------------------------------------------------------

        Arduino class initialization

        -------------------------------------------------------------------------------------
        Args:
            port:: [str]
                Arduino port address.  Can be found if Arduino IDE if unfamiliar with Terminal
                commands for finding this.

            keys_dout:: [dict]
                Should be dictionary where each key is the name for a digital output and each
                item is the pin ID.

            keys_din:: [dict]
                Should be dictionary where each key is the name for a digital output and each
                item is a tuple, corresponding to pin id and whether or not the input needs
                to be configured as input on Arduino side. Most digital inputs do. However, for inputs on 
                capacitve touch sensor, this is not the case.

            baudrate:: [unsigned integer]
                Baudrate for communicating with Arduino.
                
            timeout:: [unsigned integer]
                Millisecond timeout for Arduino during connection
                
        -------------------------------------------------------------------------------------        
        Attributes:
            ser:: [serial.Serial]
                pySerial.Serial connection for Arduino communication.
                
            keys_dout:: [dict]
                Dictionary where each key is the name for a digital output and each
                item is the pin ID.

            keys_din:: [dict
               Should be dictionary where each key is the name for a digital output and each
               item is a tuple, corresponding to pin id and whether or not the input needs
               to be configured as input on Arduino side. Most digital inputs do. However, for inputs on 
               capacitve touch sensor, this is not the case.

            input_sts:: [dict]
                Dictionary containing port of each input and corresponding state (0/1).
            
            cmnds:: [dict]
                Dictionary containing values for transmitting signals to Arduino. Note that the 
                meaning of these signals (currently 0,1,2,255) is set on the Arduino side.
            
            cmndflg:: [bytes]
                Flag indicating end of command signal in pySerial buffer. Do not change unless
                also changing Arduino/GenericSerial files.
            
            q_tasks:: [queue.Queue]
                Thread queue for holding commands until they are to be sent.

            q_inputs:: [queue.Queue}
                Thread queue for holding digital input events. Note that only changes in input
                states are logged (i.e., for 10 sec continuous input, onset and offset will be sent).
                Each item in q_inputs will contain dictionary with the following keys:
                    'din' : input port number
                    'state' : 0/1.  
                    'time' : computer timestamp of event.

            state:: [str]
                Current state of Arduino [unitiated/initiated]
        
        -------------------------------------------------------------------------------------        
        Notes:
        
        """

        self.ser = serial.Serial(port = port, baudrate = baudrate, timeout = timeout)
        self.keys_dout = keys_dout
        self.keys_din = keys_din
        self.cmnds = dict(low=0, high=1, setout=2, setin=3)
        self.cmndflg = bytes([255])
        self.input_sts = None
        self.q_tasks = queue.Queue()
        self.q_inputs = queue.Queue()
        self.state = 'uninitiated'

        
        
    def initialize(self):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Initialized connection with Arduino. Tests communication, configures inputs/outputs,
        and starts thread for transmitting commands to Arduino
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        self.handshake()
        self.io_config()
        Thread(target=self.io_transmitter, args=()).start()
        print('state: ready')
        self.state = 'ready'
       
    
    
    def stop(self):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Closes serial port and stops thread for transmitting commands to Arduino
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        if self.state=='ready':
            self.state = 'stopped'
        else:
            self.ser.close()
    
    
    
    def io_config(self):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Configures Arduino digital inputs and outputs
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        if self.keys_dout is not None:
            for name, pin in self.keys_dout.items():
                ts = time.time()
                self.io_send((pin, self.cmnds['setout']), ts)
            print('outputs configured')

        if self.keys_din is not None:
            for name, (pin, config) in self.keys_din.items():
                if config:
                    ts = time.time()
                    self.io_send((pin, self.cmnds['setin']), ts)
            self.input_sts = {x[0]: None for x in self.keys_din.values()}
            print('inputs configured')
      
    
    
    def io_transmitter(self, timeout=.001):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Controls transmission of signals to arduino
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        while True:
            cur_ts = time.time()
            hold_tasks = []
            if self.keys_din is not None and self.ser.in_waiting>2:
                input = self.ser.read_until(expected=self.cmndflg)
                input = dict(din=input[0], state=input[1], time=time.time())
                self.input_sts[input['din']] = input['state']
                self.q_inputs.put(input)     
            while not self.q_tasks.empty():
                ts, cmd = self.q_tasks.get()
                if ts <= cur_ts:
                    cmd = cmd + self.cmndflg
                    self.ser.write(cmd)
                else:
                    hold_tasks.append((ts, cmd))
            for tsk in hold_tasks:
                self.q_tasks.put(tsk)
            if self.state=='stopped':
                for name, pin in self.keys_dout.items():
                    self.digitalLow(pin)
                self.ser.close()
                break
            time.sleep(timeout)
    
    
    
    def io_send(self, sig, ts):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Adds tasks to task queue, to be executed by io_transmitter thread.
        
        -------------------------------------------------------------------------------------
        Args:
            sig:: [tuple]
                Tuple of length 2 where first index should be pin and second item is command
                (low=0, high=1, setout=2, setin=3)
                
            ts:: [float]
                Timestamp of signal
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        self.q_tasks.put((ts, bytes(sig)))
    
    
    
    def digitalHigh(self, pin, hold=None):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Sets digital output pins to high. Can be done for set period of time. 
        
        -------------------------------------------------------------------------------------
        Args:
            pin:: [int or string]
                Either an integer, specifying output pin, or a string that serving as key in
                Arduino.keys_dout
                
            hold:: [float]
                Duration, in seconds, before signal is reversed.
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        cur_ts = time.time()
        if type(pin) is str:
            pin = self.keys_dout[pin]
        self.io_send((pin, self.cmnds['high']), cur_ts)
        if hold is not None:
            self.io_send((pin, self.cmnds['low']), cur_ts + hold)
    
    
    
    def digitalLow(self, pin, hold=None):
        
        
        """ 
        -------------------------------------------------------------------------------------
        
        Sets digital output pins to low. Can be done for set period of time. 
        
        -------------------------------------------------------------------------------------
        Args:
            pin:: [int or string]
                Either an integer, specifying output pin, or a string that serving as key in
                Arduino.keys_dout
                
            hold:: [float]
                Duration, in seconds, before signal is reversed.
        
        -------------------------------------------------------------------------------------
        Notes:

        """
        
        cur_ts = time.time()
        if type(pin) is str:
            pin = self.keys_dout[pin]
        self.io_send((pin, self.cmnds['low']), cur_ts)
        if hold is not None:
            self.io_send((pin, self.cmnds['high']), cur_ts + hold)

            
            
    def handshake(self, timeout=5):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Tests for bi-directional communication with Arduino.
        
        -------------------------------------------------------------------------------------
        Args:
            timeout:: [float]
                Duration, in seconds, to wait for signal from Arduino.

        -------------------------------------------------------------------------------------
        Notes:

        """
        
        ts = time.time()
        while True:
            self.ser.write(self.cmndflg)
            data = self.ser.read()
            if data == self.cmndflg:
                print('handshake success')
                break
            if time.time()-ts > timeout:
                print('timeout without connection')
                break


    
    def flushInput(self):
        
        """ 
        -------------------------------------------------------------------------------------
        
        Flush digiital inputs and clear from self.q_tasks
        
        -------------------------------------------------------------------------------------

        Notes:


        """

        self.ser.flushInput()
        while not self.q_inputs.empty():
            self.q_inputs.get()







