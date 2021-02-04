import pandas as pd
import numpy as np
import nashpy as nash
import  matplotlib.pyplot as plt

class Game:
    type = 'PH2'
    # information
    cars = range(4, 26)
    cars = np.array(cars)
    # old thresh
    thresholdsPH2 = np.array([])
    opsPH2 = np.array([])
    target, data = [], []
    ds = []
    im = 198
    # strategies info
    sLengths = [6, 8, 5, 3, 2, 2]
    sStarts = [0, 6, 14, 19, 22, 24]
    sEnds = [6, 14, 19, 22, 24, 26]
    @staticmethod
    def load():
        '''
            load data from csv
        '''
        XData = pd.read_csv('D:/HAKIM/MIV M2/PFE/application/melanomaApp/detector/utils/resnew '+Game.type+'.csv', header=None)
        target = XData.loc[:,2].values
        data = XData.loc[:,4:29].values
        return target, data

    @staticmethod
    def getDataMatrix(s, t, minimum):
        '''
            fill the M matrix with data and (M, data, target)\n
            s is an array of methodes for a strategy s = [0, 1, ..., 21]
        '''
        results = pd.DataFrame(Game.data)
        # ABCD of melanome
        AMelanome = []
        for i in range(0, len(Game.data)):
            cols = results.loc[i].values
            A = cols[s]
            if t==1:
                carMelanome = A[
                    ((Game.opsPH2[s]==0) & (A>=Game.thresholdsPH2[s]))
                    | ((Game.opsPH2[s]==1) & (A<Game.thresholdsPH2[s]))
                ]
            else:
                carMelanome = A[
                    ((Game.opsPH2[s]==0) & (A<Game.thresholdsPH2[s]))
                    | ((Game.opsPH2[s]==1) & (A>=Game.thresholdsPH2[s]))
                ]
            if len(carMelanome) >= minimum and Game.target[i]==t:
                AMelanome.append(A)
        return AMelanome

    @staticmethod
    def getColumnsToUse(T):
        '''
            get the columns to use in each strategy methods for a sample image caracteristiques T
        '''
        cars = ((Game.opsPH2==0) & (T<Game.thresholdsPH2)) | ((Game.opsPH2==1) & (T>=Game.thresholdsPH2))
        cars = np.logical_not(cars)
        cars = np.array(cars, dtype=np.int)
        return cars

    @staticmethod
    def getColsFromStrategy(s, colsToUse):
        '''
            return columns for a strategy s
        '''
        return colsToUse[(colsToUse>=Game.sStarts[s]) & (colsToUse<Game.sEnds[s])]

    @staticmethod
    def distance(T, AMelanome, t):
        '''
            returns the distance of the caracteristiques vector T and the AMelanomes of target == t
        '''
        AMelanome = np.array(AMelanome)
        mean = np.mean(AMelanome, axis=0)
        sigma = np.std(AMelanome, axis=0, ddof=1)
        Z = np.subtract(AMelanome, mean)
        sigma[sigma==0] = 1
        Z = np.divide(Z, sigma)
        R = np.dot(Z.T, Z)
        R = np.multiply(R, 1/len(AMelanome))
        Tz = np.subtract(T, mean)
        Tz = np.divide(Tz, sigma)
        diff = np.subtract(Tz, Z)
        nn = np.linalg.norm(diff, axis=1)
        Y = np.argmin(nn, axis=0)
        Y = Z[Y]
        V = np.subtract(Tz, Y)
        d = np.dot(R, V)
        d = np.dot(V.T, d)
        if t==0:
            if d != 0:
                d = 1/d
            else:
                pass
        return d

    @staticmethod
    def Utility(d1, d2):
        '''
            Utility functions between S1 and S2
        '''
        return d2 - d1
    @staticmethod
    def getResult(T, nbStrategies):
        '''
            take a sample T
        '''
        # T = data[im]
        cars = Game.getColumnsToUse(T)
        cols = range(0, 26)
        cols = np.array(cols)
        sMelanome = cols[cars==1]
        sNonMelanome = cols[cars==0]
        ###################################
        # fill distances for player 1 (t==1)
        t = 1
        d1 = []
        strategies1 = []
        for s1 in range(0, nbStrategies):
            sMelanome1 = Game.getColsFromStrategy(s1, sMelanome)
            if len(sMelanome1)>0:
                strategies1.append(s1)
                sMelanome1 = cols[Game.sStarts[s1]:Game.sEnds[s1]]
                mins = [[6, 8, 5, 3, 1, 1], [6, 7, 2, 3, 1, 1]]
                maximum = min([len(sMelanome1), mins[t][s1]])
                M = Game.getDataMatrix(sMelanome1, t=t, minimum=maximum)
                M = np.array(M)
                # get the distance between T and M
                if(len(M)>0):
                    d = Game.distance(T[sMelanome1], M, t)
                    d1.append(d)
        d1 = np.array(d1)
        # fill distances for player 2 (t==0)
        t = 0
        d2 = []
        strategies2 = []
        for s2 in range(0, nbStrategies):
            sMelanome2 = Game.getColsFromStrategy(s2, sNonMelanome)
            if len(sMelanome2)>0:
                strategies2.append(s2)
                sMelanome2 = cols[Game.sStarts[s2]:Game.sEnds[s2]]
                mins = [[6, 8, 5, 3, 1, 1], [6, 7, 2, 3, 1, 1]]
                maximum = min([len(sMelanome2), mins[t][s2]])
                M = Game.getDataMatrix(sMelanome2, t=1, minimum=maximum)
                M = np.array(M)
                # get the distance between T and M
                if(len(M)>0):
                    d = Game.distance(T[sMelanome2], M, t)
                    d2.append(d)
        d2 = np.array(d2)
        # construct the game
        game = np.zeros((len(d1), len(d2)))
        for i in range(0, len(d1)):
            for j in range(0, len(d2)):
                game[i, j] = Game.Utility(d1[i], d2[j])
        gg = game
        game = nash.Game(game)
        equilibria = game.support_enumeration()
        if len(d1)==0:
            return 0, gg, sMelanome, sNonMelanome, (-1,-1)
        elif len(d2)==0:
            return 1, gg, sMelanome, sNonMelanome, (-1,-1)
        else:
            for en in equilibria:
                ii = np.argmax(en[0])
                jj = np.argmax(en[1])
                result = 1 if gg[ii, jj]>=0 else 0
                return result, gg, sMelanome, sNonMelanome, (ii,jj)
                break
        return -1, gg, sMelanome, sNonMelanome, (-1,-1)

    @staticmethod
    def init(type=None):
        if type!=None:
            Game.type = type
        # load data
        if(Game.type=='PH2'):
            Game.thresholdsPH2 = np.array([2.65, 92.87, 6.39, 13.2, 17.2, 15.44, 55.73, 1560, 0.02, 0.56, 1.81, 1.35, 219, 1, 5, 2, 5, 9.51, 63.69, 560, 572.24, 4.54, 1, 1, 6.11, 0.01])
            Game.opsPH2 = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        else:            
            Game.thresholdsPH2 = np.array([4.23, 93.61, 7.31, 12.28, 16.17, 10.18, 73.42, 900, 0.02, 0.71, 1.37, 1.2, 145, 1.6, 3, 2, 3, 10.25, 66.93, 342, 323.27, 3.63, 0, 0, 0.05, 0])
            Game.opsPH2 = np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        Game.target, Game.data = Game.load()