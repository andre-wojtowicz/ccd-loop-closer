#!/usr/bin/env python
"""
Cyclic Coordinate Descent algorithm.
On the basis of an article 'Cyclic coordinate descent: A robotics algorithm
for protein loop closure.' - Canutescu AA, Dunbrack RL Jr.
"""

from numpy import (array, sin, cos, arctan2, dot, cross, pi, sqrt, diag,
                   insert, delete)
from numpy.linalg import inv, norm
from numpy.random import random

PRECISION = 4

class CCDError(Exception):
    pass


class CCDLoopCloser:
    """
    Class for running the CCD loop closure algorithm
    on a list of fixed and a set of moved atoms.
    """

    def __init__(self, chain, fixed):
        """
        chain - list of 3+ atoms that are moved.
        fixed - list of target three atoms (C-Anchor).
        All atoms should be numpy.array objects.
        """

        # initial check of input corectness
        if len(chain) < 3:
            return "ERR", "chain length must equal at least 3"

        if len(fixed) != 3:
            return "ERR", "fixed length must equal to 3"

        self.chain = chain[:]
        self.fixed = fixed[:]


    @classmethod
    def arbitrary_rotate(self, pntM, pntO, rot_angle, uvecTheta):
        """
        3D rotation of a point.
        pntM - moving point.
        pntO - central point of rotation.
        rot_angle - rotation angle (radian).
        uvecTheta - unit vector of y in local coordinate system.
        """
        
        vecR = pntM - pntO   

        try:
            uvecR = self.normalise(vecR)
        except CCDError:
            # don't rotate when the point is on rotation axis
            return pntM
            
        uvecS = cross(uvecR, uvecTheta)
        
        return (norm(vecR) * cos(rot_angle) * uvecR +
                norm(vecR) * sin(rot_angle) * uvecS) + pntO


    def calc_rmsd(self):
        """
        Returns RMSD fit of last 3 moving vectors to fixed vectors.
        """
        
        rmsd = 0.0
        for i in range(1, 4):
            vec = self.chain[-i] - self.fixed[-i]
            rmsd += dot(vec, vec)

        return sqrt(rmsd/3.0)


    @staticmethod
    def get_rotation_axis(pntP, pntQ):
        """
        Calculates parameters of a line crossing 2 fixed points
        in 3D in following form:

        pntP = [x_1, y_1, z_1]
        pntQ = [x_2, y_2, z_2]


         x  - x_1    y  - y_1    z - z_1
        --------- = --------- = --------- = t
        x_2 - x_1   y_2 - y_1   z_2 - z_1


        x - x_1   y - y_1  z - z_1
        ------- = ------ = ------- = t
           a         b        c
        """

        params = {}

        params['x_1'] = float(pntP[0])
        params['y_1'] = float(pntP[1])
        params['z_1'] = float(pntP[2])
        params['a']   = float(pntQ[0] - pntP[0])
        params['b']   = float(pntQ[1] - pntP[1])
        params['c']   = float(pntQ[2] - pntP[2])

        return params


    @staticmethod
    def get_rotation_central_point(line, pntP, pntQ, pntM_Ox):
        """
        Computes coordinates of a pntO_x which lies on the line
        and vector /pntQ pntP/ is orthogonal to vector
        /pntO_x pntM_Ox/ (dot product of /pntQ pntP/
        and /pntO_x pntM_Ox/ equals to zero).
        Parameters of the line are obtained from pntP and pntQ.
        """

        vecQP = pntP - pntQ

        line_xyz = array([line['x_1'], line['y_1'], line['z_1']])
        line_abc = array([line['a'], line['b'], line['c']])

        t_numerator = dot(vecQP, pntM_Ox) - dot(vecQP, line_xyz)
        t_denominator = dot(vecQP, line_abc)

        t = t_numerator / float(t_denominator)

        pntO_x = array([
            line['x_1'] + line['a'] * t,
            line['y_1'] + line['b'] * t,
            line['z_1'] + line['c'] * t,
        ])

        return pntO_x


    @staticmethod
    def is_on_rotation_axis(pnt, axis):
        """
        Checks if a point lies on an axis.
        axis - dict of parameters
        """
        
        if round(axis['a'], PRECISION) != 0.0:
            t = (pnt[0] - axis['x_1']) / axis['a']
        elif round(axis['b'], PRECISION) != 0.0:
            t = (pnt[1] - axis['y_1']) / axis['b']
        elif round(axis['c'], PRECISION) != 0.0:
            t = (pnt[2] - axis['z_1']) / axis['c']
        else:
            t = 0.0

        if (pnt[0] == axis['x_1'] + axis['a'] * t
            and pnt[1] == axis['y_1'] + axis['b'] * t
            and pnt[2] == axis['z_1'] + axis['c'] * t):
            return True
        else:
            return False


    @staticmethod
    def normalise(vec):
        """
        Normalises vector.
        """

        nrm = norm(vec)

        if nrm == 0.0:
            raise CCDError('Normalisation error; vector length eq 0')
        else:
            return vec / nrm


    def run_ccd(self, threshold=0.1, max_it=5000):
        """
        The moving vectors are changed until its last three elements
        overlap with the fixed ones with a RMSD smaller than threshold.
        """
        
        iteration = 0
        while True:

            # check whether desired rmsd achieved
            rmsd = self.calc_rmsd()

            if rmsd < threshold:
                return "SUCC", rmsd, iteration

            #if iteration % 100 == 0:
            #    print iteration, rmsd
            
            # TODO: check if algorithm stuck in local minimum

            # check if executed maximum number of iterations
            if iteration == max_it:
                return "MAX_IT", rmsd, iteration

            # for almost each edge find best rotation angle
            for i in range(len(self.chain)-2):
                # ------------------- to_del?
                #rmsd = self.calc_rmsd()
                #if rmsd < threshold:
                #    return "SUCC", rmsd, iteration
                # ------------------- end to_del?

                # determine rotation axis
                pntN = self.chain[i]
                pntC_alpha = self.chain[i+1]

                rot_axis = self.get_rotation_axis(pntN, pntC_alpha)

                # determine unit vector theta
                vecNC_aplha = pntC_alpha - pntN
                
                uvecTheta = self.normalise(vecNC_aplha)

                # coefficients of S
                a = 0.0
                b = 0.0
                c = 0.0

                # for moving and fixed C-terminal anchors...
                for j in range(1, 4):

                    pntM_Oj = self.chain[-j]       
                    pntF_j = self.fixed[-j]

                    # find point O_j
                    pntO_j = self.get_rotation_central_point(rot_axis,
                        pntN, pntC_alpha, pntM_Oj)

                    vecR_j = pntM_Oj - pntO_j
                    vecF_j = pntF_j - pntO_j

                    # skip if moving atom is on rotation axis
                    if self.is_on_rotation_axis(pntM_Oj, rot_axis):
                        continue

                    # determine unit vectors (r and s) for local
                    # coordinate system
                    uvecR_j = self.normalise(vecR_j)

                    uvecS_j = cross(uvecR_j, uvecTheta)

                    # calculate one element of a: (r_j)^2 + (f_j)^2
                    r_j = norm(vecR_j)
                    f_j = norm(vecF_j)

                    # if fact it's not necessary to get a
                    a += r_j**2 + f_j**2

                    # calculate one element of b: 2 * r_j *
                    # dot(vecF_j, uvecR_j)
                    b += 2 * r_j * dot(vecF_j, uvecR_j)

                    # calculate one element of c: 2 * r_j *
                    # dot(vecF_j, uvecS_j)
                    c += 2 * r_j * dot(vecF_j, uvecS_j)


                # find rotation angle
                cos_alpha = b / sqrt(b**2 + c**2)
                sin_alpha = c / sqrt(b**2 + c**2)
                rot_angle = arctan2(sin_alpha, cos_alpha)

                # apply rotation for next points
                for j in range(i+2, len(self.chain)):
                    pntO = self.get_rotation_central_point(rot_axis,
                        pntN, pntC_alpha, self.chain[j])
                    self.chain[j] = self.arbitrary_rotate(self.chain[j], pntO,
                        rot_angle, uvecTheta)

                    # eliminate numeric precision error
                    for k in range(3):
                        self.chain[j][k] = round(self.chain[j][k], PRECISION)

            iteration += 1


if __name__ == '__main__':
    
    CHAIN1 = [array([0.0, 0.0, 0.0]),
            array([3.0, 0.0, 0.0]),
            array([3.0, -2.0, 0.0]),
            array([6.0, -2.0, 0.0]),
            array([6.0, 0.0, 0.0]),
            array([8.0, 0.0, 0.0]),
            array([8.0, 2.0, 0.0])
            ]
    FIXED1 = [array([6.0, 0.0, 0.0]),
            array([8.0, 0.0, 0.0]),
            array([8.0, -2.0, 0.0])
            ]
    
    ALG1 = CCDLoopCloser(CHAIN1, FIXED1)
    print(ALG1.run_ccd())

    #--------------------------------------------

    MAXIT = 250
    LOOPS = 500
    CHAIN_LEN = 6

    SUCC = 0

    print("loop\teff. so far\tresult\trmsd\titerations")

    for n in range(LOOPS):
        PNT = array([0.0, 0.0, 0.0])
        CHAIN2 = [PNT]

        for j in range(CHAIN_LEN):
            NEXT_PNT = random(3)
            NEXT_PNT = NEXT_PNT/norm(NEXT_PNT)
            NEXT_PNT = PNT + NEXT_PNT*2.6
            CHAIN2.append(NEXT_PNT)
            PNT = NEXT_PNT

        FIXED2 = CHAIN2[-3:]
            
        for j in range(-5, -2):
            PHI = 2*pi*random()
            ROT_AXIS = CCDLoopCloser.get_rotation_axis(CHAIN2[j], CHAIN2[j+1])
            VECPQ = CHAIN2[j+1] - CHAIN2[j]
            UVECTHETA = VECPQ / norm(VECPQ)
            
            for k in range(j+2, 0):
                PNTO = CCDLoopCloser.get_rotation_central_point(ROT_AXIS,
                    CHAIN2[j], CHAIN2[j+1], CHAIN2[k])
                CHAIN2[k] = CCDLoopCloser.arbitrary_rotate(CHAIN2[k], PNTO,
                    PHI, UVECTHETA)

        ALG2 = CCDLoopCloser(CHAIN2, FIXED2)
        RES, RMSD, ITER = ALG2.run_ccd(max_it=MAXIT)
        if RES == 'SUCC':
            SUCC += 1
        print("{0}\t{1}\t\t{2}\t{3}\t{4}".format(n,
            str(round(SUCC*100/float(n+1),2))+'%', RES,
            str(round(RMSD, PRECISION)), ITER))
