from dataclasses import dataclass, field
from typing import List
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

from sympy.polys.numberfields.subfield import field_isomorphism_factor

length = 201  # board length (x)
width = 201  # board width (y)
height = 201  # board height (z)

dt = 1 * (10 ** -3)
dx = 5 * (10 ** -3)
dy = 5 * (10 ** -3)
# Mesenchymal-like cancer cell diffusion coeff
D_M = 1 * (10 ** -4)
# Epithelial-like cancer cell diffusion coeff
D_E = 5 * (10 ** -5)
# Mesenchymal haptotatic sensitivity coeff
phi_M = 5 * (10 ** -4)
# Epithelial haptotatic sensitivity coeff
phi_E = 5 * (10 ** -4)
# MMP-2 diffusion coefficient
D_m = 1 * (10 ** -3)
# MMP-2 production rate
theta = 0.195
# MMP-2 decay rate
MMP2_decay = 0.1
# ECM degradation rate by MT1-MMP
gamma_1 = 1
# ECM degradation rate by MMP-2
gamma_2 = 1
# Time CTC's spend in vasculature
T_v = 0.18
# epithelial doubling time
T_E = 3
# mesenchymal doubling time
T_M = 2
# single CTC survival probability
P_s = 5 * (10 ** -4)
# cluster survival probability
P_C = 2.5 * (10 ** -4)
# extravasion prob to bones
E_1 = 0.5461
# extravasion prob to lungs
E_2 = 0.2553
# extravasion prob to liver
E_3 = 0.1986
# Number of iterations to run
ITERATIONS = 10
# Max steps in vasaclature
vasc_time = 6
# Probability cluster disaggregates
P_d = .3  # idk tbh the paper doesn't say...



@dataclass
class PrimaryGrid:
    """
    Represents 201 x 201 x 201 primary and secondary grids
    -5 dictionaries for mesenchymal, epithelial, MM2, ECM concentration, blood vessels (with keeping track of normal or ruptured)
    -key = location (tuple)
    -value = concentration (int, how many of them there are)
    -Specifc PDES (methods)
    """
    mes: dict = field(default_factory=dict)  # mesenchymal
    epi: dict = field(default_factory=dict)  # epithelial
    MMP2: dict = field(default_factory=dict)  # matrix metalloproteinase-2
    ECM: dict = field(default_factory=dict)  # extracellular matrix
    bv: dict = field(default_factory=dict)  # blood vessels
    clusters: List[tuple] = field(default_factory=list)  # clusters leaving
    time : int = 0 

    def preview(self, ax, z_slice = None):
        #show a central z slice of the grid
        if z_slice is None:
            z_slice = height // 2
        grid = []
        for x in range(length):
            grid_row = []
            for y in range(width):
                    # Replace this with whichever you'd like to preview
                    grid_row.append(self.bv.get((x, y, z_slice), 0))
            grid.append(grid_row)
        
        ax.clear()
        ax.set_title(f'Z slice at {z_slice}')
        ax.imshow(grid)

    # Rishika
    def initialize_primary(self):
        """
        adds mesechmmal cancer cells cluster in middle
        add epi
        """
        # I made my boundry flux thing assuming zero based indicies - Mia
        # find distances for all 201 points from center
        dist_points = {}
        dist_list = []
        center = (length//2, width//2, height//2)
        for x in range(2, length - 2):
            for y in range(2, width - 2):
                for z in range(2, height - 2):
                    distance = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
                    dist_points[(x, y, z)] = distance
                    dist_list.append(distance)
        dist_list.sort()
        # sort and use everything outside of first 200 in [2, 198] for 10 blood vessels
        twohundreth = dist_list[199]
        ninetyseventh = dist_list[96]
        min_coord, max_coord = 2, 198
        minz, maxz = 2, height - 3

        # use outer 200 for the 10 blood vessels (5 ruptured, 5 normal)
        i = 0
        while i < 10:
            random_x = random.randint(min_coord, max_coord)
            random_y = random.randint(min_coord, max_coord)
            random_z = random.randint(minz, maxz)
            
            if (random_x, random_y, random_z) in self.bv or dist_points[(random_x, random_y, random_z)] < twohundreth:
                continue
            if i < 2:
                self.bv[(random_x, random_y, random_z)] = 5
                self.bv[(random_x-1, random_y, random_z)] = 5
                self.bv[(random_x+1, random_y, random_z)] = 5
                self.bv[(random_x, random_y-1, random_z)] = 5
                self.bv[(random_x, random_y+1, random_z)] = 5
                self.bv[(random_x, random_y, random_z-1)] = 5
                self.bv[(random_x, random_y, random_z+1)] = 5
            else:
                self.bv[(random_x, random_y, random_z)] = 1
            i += 1
        
        # use middle 97 for the 388 cancer cells 
        i = 0
        while i < 388:
            random_x = random.randint(min_coord, max_coord)
            random_y = random.randint(min_coord, max_coord)
            random_z = random.randint(minz, maxz)
            if dist_points[(random_x, random_y, random_z)] > ninetyseventh:
                continue
            
            # first 40% are epithelial-like phenotype, and next 60% of mesenchymal-like phenotype
            epi = self.epi.get((random_x, random_y, random_z), 0)
            if i < 155:
                if epi == 4:
                    continue
                else:
                    self.epi[(random_x, random_y, random_z)] = epi + 1
                    i+=1
            else:
                mes = self.mes.get((random_x, random_y, random_z), 0)
                if epi+mes == 4:
                    continue
                else:
                    self.mes[(random_x, random_y, random_z)] = mes + 1
                    i+=1

    # Carmen
    def update_MMP2(self) -> None:
        #m_new = m + dt*(Dm*laplacian(m) + theta*cm - lambda*m)
        
        new = {}

        for x in range(length):
            for y in range(width):
                for z in range(height):
                    currMMP2 = self.MMP2.get((x, y, z), 0)
                    currMes = self.mes.get((x, y, z), 0)

                    #laplacian(m) = (m_x+1,y + m_x-1,y + m_x,y+1 + m_x,y-1 - 4*m_xy)/dx^2
                    up = self.MMP2.get((min(length-1, x+1), y, z), 0)
                    down = self.MMP2.get((max(0, x-1), y, z), 0)
                    right = self.MMP2.get((x, min(y+1, width-1), z), 0)
                    left = self.MMP2.get((x, max(0, y-1), z), 0)
                    front = self.MMP2.get((x, y, min(z+1, height-1)), 0)
                    back = self.MMP2.get((x, y, max(0, z-1)), 0)

                    lap_m = (up + down + right + left + front + back - 6*currMMP2) / (dx**2)
                    dm_dt = D_m * lap_m + theta*currMes - MMP2_decay*currMMP2
                    newMMP2 = currMMP2 + dt*dm_dt

                    new[(x, y, z)] = newMMP2

        self.MMP2 = new

    # Carmen
    def update_ECM(self) -> None:
        #new_w = w + dt*-(gamma1*cm + gamma2*m)*w
    
        new = {}

        for x in range(length):
            for y in range(width):
                for z in range(height):
                    currECM = self.ECM.get((x, y, z), 1)
                    currMMP2 = self.MMP2.get((x, y, z), 0)
                    currMes = self.mes.get((x, y, z), 0)

                    dw_dt = -(gamma_1*currMes + gamma_2*currMMP2)*currECM
                    newECM = currECM + dt*dw_dt

                    new[(x, y, z)] = newECM
        
        self.ECM = new

    # Mia
    def update_mesechymal_movement(self) -> None:
        """
        simulate movement
        simulate mitosis
        """
        new_mes = (self.mes).copy()
        # the order we do this does have effect as we test capacity as we update
        for (x, y, z), concentration in self.mes.items():
            for _ in range(0, concentration):
                ECM_conc_left = self.ECM[(x-1,y,z)] if (x-1, y, z) in self.ECM else 1
                ECM_conc_right = self.ECM[(x+1,y,z)] if (x+1, y, z) in self.ECM else 1
                ECM_conc_down = self.ECM[(x,y-1,z)] if (x, y-1, z) in self.ECM else 1
                ECM_conc_up = self.ECM[(x,y+1,z)]  if (x, y+1, z) in self.ECM else 1
                ECM_conc_front = self.ECM[(x,y,z+1)]  if (x, y, z+1) in self.ECM else 1
                ECM_conc_back = self.ECM[(x,y,z-1)]  if (x, y, z-1) in self.ECM else 1

                r = random.random()
                prob_move_left = (dt / (dx ** 2)) * (D_M -
                                  (phi_M/6) * (ECM_conc_right - ECM_conc_left))
                prob_move_right = (dt / (dx ** 2)) * (D_M +
                                   (phi_M/6) * (ECM_conc_right - ECM_conc_left))
                prob_move_down = (dt / (dx ** 2)) * (D_M +
                                  (phi_M/6) * (ECM_conc_up - ECM_conc_down))
                prob_move_up = (dt / (dx ** 2)) * (D_M - 
                                    (phi_M/6) * (ECM_conc_up - ECM_conc_down))
                prob_move_front = (dt / (dx**2)) * (D_M + 
                                    (phi_M/6) * (ECM_conc_front - ECM_conc_back))
                prob_move_back = (dt / (dx**2)) * (D_M - 
                                    (phi_M/6) * (ECM_conc_front - ECM_conc_back))
                
                if r < prob_move_left:  # cell moves left
                    if x > 0 and new_mes.get((x-1, y, z), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x-1, y, z)] = new_mes[(x-1, y, z)] + 1 if (x-1, y, z) in new_mes else 1
                    else:
                        continue  # no change if on left boundry or capacity already reached
                elif r < prob_move_right + prob_move_left:  # cell moves right
                    if x < 200 and new_mes.get((x+1, y, z), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x+1, y, z)] = new_mes[(x+1, y, z)] + 1 if (x+1, y) in new_mes else 1
                    else:
                        continue  # no change if on right boundry or capacity already reached
                elif r < prob_move_down + prob_move_right + prob_move_left:  # cell moves down
                    if y > 0 and new_mes.get((x, y-1, z), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x, y-1, z)] = new_mes[(x, y-1, z)] + 1 if (x, y-1, z) in new_mes else 1
                    else:
                        continue  # no change if on lower boundry or capacity already reached
                elif r < prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves up
                    if y < 200 and new_mes.get((x, y+1, z), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x, y+1, z)] = new_mes[(x, y+1, z)] + 1 if (x, y+1, z) in new_mes else 1
                    else:
                        continue  # no change if on upper boundry or capacity already reached
                    
                elif r < prob_move_back + prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves back
                    if z > 0 and new_mes.get((x, y, z-1), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x, y, z-1)] = new_mes[(x, y, z-1)] + 1 if (x, y, z-1) in new_mes else 1
                    else:
                        continue  # no change if on back boundry or capacity already reached
                elif r < prob_move_front + prob_move_back + prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves front
                    if z < 200 and new_mes.get((x, y, z+1), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x, y, z+1)] = new_mes[(x, y, z+1)] + 1 if (x, y, z+1) in new_mes else 1
                    else:
                        continue  # no change if on front boundry or capacity already reached
        self.mes = new_mes

    def update_mesechymal_mitosis(self) -> None:
        self.time +=dt 
        if self.time % T_M == 0: 
            new_mes = (self.mes).copy()
            for (x, y, z), concentration in self.mes.items():
                new_mes[(x, y, z)] = concentration * 2 if concentration <= 4 else 4
            self.mes = new_mes

    # Mia

    def update_epithelial_movement(self) -> None:
        """
        simulate movement
        simulate mitosis
        """
        new_epi = (self.epi).copy()
        for (x, y, z), concentration in self.epi.items():
            for _ in range(0, concentration):
                ECM_conc_left = self.ECM[(x-1,y,z)] if (x-1, y, z) in self.ECM else 1
                ECM_conc_right = self.ECM[(x+1,y,z)] if (x+1, y, z) in self.ECM else 1
                ECM_conc_down = self.ECM[(x,y-1,z)] if (x, y-1, z) in self.ECM else 1
                ECM_conc_up = self.ECM[(x,y+1,z)]  if (x, y+1, z) in self.ECM else 1
                ECM_conc_front = self.ECM[(x,y,z+1)]  if (x, y, z+1) in self.ECM else 1
                ECM_conc_back = self.ECM[(x,y,z-1)]  if (x, y, z-1) in self.ECM else 1

                r = random.random()
                prob_move_left = (dt / (dx ** 2)) * (D_E -
                                  (phi_E/6) * (ECM_conc_right - ECM_conc_left))
                prob_move_right = (dt / (dx ** 2)) * (D_E +
                                   (phi_E/6) * (ECM_conc_right - ECM_conc_left))
                prob_move_down = (dt / (dx ** 2)) * (D_E +
                                  (phi_E/6) * (ECM_conc_up - ECM_conc_down))
                prob_move_up = (dt / (dx ** 2)) * (D_E - 
                                    (phi_E/6) * (ECM_conc_up - ECM_conc_down))
                prob_move_front = (dt / (dx**2)) * (D_E + 
                                    (phi_E/6) * (ECM_conc_front - ECM_conc_back))
                prob_move_back = (dt / (dx**2)) * (D_E - 
                                    (phi_E/6) * (ECM_conc_front - ECM_conc_back))
                
                if r < prob_move_left:  # cell moves left
                    if x > 0 and new_epi.get((x-1, y, z), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x-1, y, z)] = new_epi[(x-1, y, z)] + 1 if (x-1, y, z) in new_epi else 1
                    else:
                        continue  # no change if on left boundry or capacity already reached
                elif r < prob_move_right + prob_move_left:  # cell moves right
                    if x < 200 and new_epi.get((x+1, y, z), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x+1, y, z)] = new_epi[(x+1, y, z)] + 1 if (x+1, y, z) in new_epi else 1
                    else:
                        continue  # no change if on right boundry or capacity already reached
                elif r < prob_move_down + prob_move_right + prob_move_left:  # cell moves down
                    if y > 0 and new_epi.get((x, y-1, z), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x, y-1, z)] = new_epi[(x, y-1, z)] + 1 if (x, y-1, z) in new_epi else 1
                    else:
                        continue  # no change if on lower boundry or capacity already reached
                elif r < prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves up
                    if y < 200 and new_epi.get((x, y+1, z), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x, y+1, z)] = new_epi[(x, y+1, z)] + 1 if (x, y+1, z) in new_epi else 1
                    else:
                        continue  # no change if on upper boundry or capacity already reached
                    
                elif r < prob_move_back + prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves back
                    if z > 0 and new_epi.get((x, y, z-1), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x, y, z-1)] = new_epi[(x, y, z-1)] + 1 if (x, y, z-1) in new_epi else 1
                    else:
                        continue  # no change if on back boundry or capacity already reached
                elif r < prob_move_front + prob_move_back + prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves front
                    if z < height - 1 and new_epi.get((x, y, z+1), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x, y, z+1)] = new_epi[(x, y, z+1)] + 1 if (x, y, z+1) in new_epi else 1
                    else:
                        continue  # no change if on front boundry or capacity already reached
        self.epi = new_epi

    def update_epithelial_mitosis(self) -> None:
        self.time +=dt 
        if self.time % T_E == 0: 
            new_epi = (self.epi).copy()
            for (x, y, z), concentration in self.epi.items():
                new_epi[(x, y, z)] = concentration * 2 if concentration <= 4 else 4
            self.epi = new_epi
    
    def find_intravasating_clusters(self):
        # List: (mes count, epi count)
        clusters = []

        mes = self.mes
        epi = self.epi

        def neighbors4(x, y, z):
            out = []
            if x > 0: out.append((x - 1, y, z))
            if x < 200: out.append((x + 1, y, z))
            if y > 0: out.append((x, y - 1, z))
            if y < 200: out.append((x, y + 1, z))
            if z > 0: out.append((x, y, z - 1))
            if z < height - 1: out.append((x, y, z + 1))
            return out

        # Set of visited clusters
        visited = set()
        for (x, y, z), vessel_type in self.bv.items():
            # No vessel there, skip
            if vessel_type not in (1, 2):
                continue

            num_mes = mes.get((x, y, z), 0)
            num_epi = epi.get((x, y, z), 0)
            to_collect = []

            if vessel_type == 1:
                if num_mes == 0:
                    continue
            
                to_collect = [(x, y, z) + neighbors4(x, y, z)]
            elif vessel_type == 2:
                if num_mes == 0 and num_epi == 0:
                    continue
            
                to_collect = [(x, y, z)] + neighbors4(x, y, z)
            
            total_mes = 0
            total_epi = 0

            for loc in to_collect:
                # Location's been visited already, skip
                if loc in visited:
                    continue
                visited.add(loc)

                total_mes += mes.get(loc, 0)
                total_epi += epi.get(loc, 0)
            
            for loc in to_collect:
                if loc in mes:
                    del mes[loc]
                if loc in epi:
                    del epi[loc]
            
            if total_mes + total_epi > 0:
                clusters.append((total_mes, total_epi, 0))
            
        return clusters

    # Sarah
    def update_all(self) -> None:  # list of cells that move into vasculature
        """
        update MMP2 then ECM then mesechymal then epithelial
        figures out what cells are on blood vessels and removes them from dicts
        adds clusters leaving to self.cluster
        """
        self.update_MMP2()
        self.update_ECM()
        self.update_mesechymal_movement()
        self.update_epithelial_movement()

        new_clusters = self.find_intravasating_clusters()

        self.clusters.extend(new_clusters)
        self.update_mesechymal_mitosis()
        self.update_epithelial_mitosis()




class SecondaryGrid:
    """
    Represents 201 x 201 primary and secondary grids
    -5 dictionaries for mesenchymal, epithelial, MM2, ECM concentration, blood vessels)
    -key = location (tuple)
    -value = concentration (int, how many of them there are)
    -Specifc PDES (methods)
    """
    mes: dict = dict()  # mesenchymal
    epi: dict = dict()  # epithelial
    MMP2: dict = dict()  # matrix metalloproteinase-2
    ECM: dict = dict()  # extracellular matrix
    bv: dict = dict()  # blood vessels
    clusters: List[tuple] = []  # clusters leaving
    time : int = 0 

    # Rishika
    def initialize_secondary(self):
        """
        adds normal vessels to grid
        """
        min_coord, max_coord = 2, 198
        i = 0
        while i < 10:
            random_x = random.randint(min_coord, max_coord)
            random_y = random.randint(min_coord, max_coord)
            random_z = random.randint(min_coord, height - 3)
            if (random_x, random_y, random_z) in self.bv:
                continue
            self.bv[(random_x, random_y, random_z)] = 1
            i += 1

    # Carmen
    def update_MMP2(self) -> None:
        #m_new = m + dt*(Dm*laplacian(m) + theta*cm - lambda*m)
        
        new = {}

        for x in range(length):
            for y in range(width):
                for z in range(height):
                    currMMP2 = self.MMP2.get((x, y, z), 0)
                    currMes = self.mes.get((x, y, z), 0)

                    #laplacian(m) = (m_x+1,y + m_x-1,y + m_x,y+1 + m_x,y-1 - 4*m_xy)/dx^2
                    up = self.MMP2.get((min(length-1, x+1), y, z), 0)
                    down = self.MMP2.get((max(0, x-1), y, z), 0)
                    right = self.MMP2.get((x, min(y+1, width-1), z), 0)
                    left = self.MMP2.get((x, max(0, y-1), z), 0)
                    front = self.MMP2.get((x, y, min(z+1, height-1)), 0)
                    back = self.MMP2.get((x, y, max(0, z-1)), 0)

                    lap_m = (up + down + right + left + front + back - 6*currMMP2) / (dx**2)
                    dm_dt = D_m * lap_m + theta*currMes - MMP2_decay*currMMP2
                    newMMP2 = currMMP2 + dt*dm_dt

                    new[(x, y, z)] = newMMP2

        self.MMP2 = new

    # Carmen
    def update_ECM(self) -> None:
        #new_w = w + dt*-(gamma1*cm + gamma2*m)*w
    
        new = {}

        for x in range(length):
            for y in range(width):
                for z in range(height):
                    currECM = self.ECM.get((x, y, z), 1)
                    currMMP2 = self.MMP2.get((x, y, z), 0)
                    currMes = self.mes.get((x, y, z), 0)

                    dw_dt = -(gamma_1*currMes + gamma_2*currMMP2)*currECM
                    newECM = currECM + dt*dw_dt

                    new[(x, y, z)] = newECM
        
        self.ECM = new

    # Mia
    def update_mesechymal_movement(self) -> None:
        """
        simulate movement
        simulate mitosis
        """
        new_mes = (self.mes).copy()
        # the order we do this does have effect as we test capacity as we update
        for (x, y, z), concentration in self.mes.items():
            for _ in range(0, concentration):
                ECM_conc_left = self.ECM[(x-1,y,z)] if (x-1, y, z) in self.ECM else 1
                ECM_conc_right = self.ECM[(x+1,y,z)] if (x+1, y, z) in self.ECM else 1
                ECM_conc_down = self.ECM[(x,y-1,z)] if (x, y-1, z) in self.ECM else 1
                ECM_conc_up = self.ECM[(x,y+1,z)]  if (x, y+1, z) in self.ECM else 1
                ECM_conc_front = self.ECM[(x,y,z+1)]  if (x, y, z+1) in self.ECM else 1
                ECM_conc_back = self.ECM[(x,y,z-1)]  if (x, y, z-1) in self.ECM else 1

                r = random.random()
                prob_move_left = (dt / (dx ** 2)) * (D_M -
                                  (phi_M/6) * (ECM_conc_right - ECM_conc_left))
                prob_move_right = (dt / (dx ** 2)) * (D_M +
                                   (phi_M/6) * (ECM_conc_right - ECM_conc_left))
                prob_move_down = (dt / (dx ** 2)) * (D_M +
                                  (phi_M/6) * (ECM_conc_up - ECM_conc_down))
                prob_move_up = (dt / (dx ** 2)) * (D_M - 
                                    (phi_M/6) * (ECM_conc_up - ECM_conc_down))
                prob_move_front = (dt / (dx**2)) * (D_M + 
                                    (phi_M/6) * (ECM_conc_front - ECM_conc_back))
                prob_move_back = (dt / (dx**2)) * (D_M - 
                                    (phi_M/6) * (ECM_conc_front - ECM_conc_back))
                
                if r < prob_move_left:  # cell moves left
                    if x > 0 and new_mes.get((x-1, y, z), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x-1, y, z)] = new_mes[(x-1, y, z)] + 1 if (x-1, y, z) in new_mes else 1
                    else:
                        continue  # no change if on left boundry or capacity already reached
                elif r < prob_move_right + prob_move_left:#cell moves right
                    if x < 200 and new_mes.get((x+1, y, z), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x+1, y, z)] = new_mes[(x+1, y, z)] + 1 if (x+1, y, z) in new_mes else 1
                    else:
                        continue  # no change if on right boundry or capacity already reached
                elif r < prob_move_down + prob_move_right + prob_move_left:  # cell moves down
                    if y > 0 and new_mes.get((x, y-1, z), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x, y-1, z)] = new_mes[(x, y-1, z)] + 1 if (x, y-1, z) in new_mes else 1
                    else:
                        continue  # no change if on lower boundry or capacity already reached
                elif r < prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves up
                    if y < 200 and new_mes.get((x, y+1, z), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x, y+1, z)] = new_mes[(x, y+1, z)] + 1 if (x, y+1, z) in new_mes else 1
                    else:
                        continue  # no change if on upper boundry or capacity already reached
                    
                elif r < prob_move_back + prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves back
                    if z > 0 and new_mes.get((x, y, z-1), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x, y, z-1)] = new_mes[(x, y, z-1)] + 1 if (x, y, z-1) in new_mes else 1
                    else:
                        continue  # no change if on back boundry or capacity already reached
                elif r < prob_move_front + prob_move_back + prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves front
                    if z < 200 and new_mes.get((x, y, z+1), 0) <= 3:
                        new_mes[(x, y, z)] = new_mes[(x, y, z)] - 1
                        new_mes[(x, y, z+1)] = new_mes[(x, y, z+1)] + 1 if (x, y, z+1) in new_mes else 1
                    else:
                        continue  # no change if on front boundry or capacity already reached
        self.mes = new_mes     

    def update_mesechymal_mitosis(self) -> None:
        self.time+=dt
        if self.time % T_M == 0: 
            new_mes = (self.mes).copy()
            for (x, y, z), concentration in self.mes.items():
                new_mes[(x, y, z)] = concentration * 2 if concentration * 2 <= 4 else 4
            self.mes = new_mes

    #Mia
    def update_epithelial_movement(self) -> None:
        """
        simulate movement
        simulate mitosis
        """
        new_epi = (self.epi).copy()
        for (x, y, z), concentration in self.epi.items():
            for _ in range(0, concentration):
                ECM_conc_left = self.ECM[(x-1,y,z)] if (x-1, y, z) in self.ECM else 1
                ECM_conc_right = self.ECM[(x+1,y,z)] if (x+1, y, z) in self.ECM else 1
                ECM_conc_down = self.ECM[(x,y-1,z)] if (x, y-1, z) in self.ECM else 1
                ECM_conc_up = self.ECM[(x,y+1,z)]  if (x, y+1, z) in self.ECM else 1
                ECM_conc_front = self.ECM[(x,y,z+1)]  if (x, y, z+1) in self.ECM else 1
                ECM_conc_back = self.ECM[(x,y,z-1)]  if (x, y, z-1) in self.ECM else 1

                r = random.random()
                prob_move_left = (dt / (dx ** 2)) * (D_E -
                                  (phi_E/6) * (ECM_conc_right - ECM_conc_left))
                prob_move_right = (dt / (dx ** 2)) * (D_E +
                                   (phi_E/6) * (ECM_conc_right - ECM_conc_left))
                prob_move_down = (dt / (dx ** 2)) * (D_E +
                                  (phi_E/6) * (ECM_conc_up - ECM_conc_down))
                prob_move_up = (dt / (dx ** 2)) * (D_E - 
                                    (phi_E/6) * (ECM_conc_up - ECM_conc_down))
                prob_move_front = (dt / (dx**2)) * (D_E + 
                                    (phi_E/6) * (ECM_conc_front - ECM_conc_back))
                prob_move_back = (dt / (dx**2)) * (D_E - 
                                    (phi_E/6) * (ECM_conc_front - ECM_conc_back))
                
                if r < prob_move_left:  # cell moves left
                    if x > 0 and new_epi.get((x-1, y, z), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x-1, y, z)] = new_epi[(x-1, y, z)] + 1 if (x-1, y, z) in new_epi else 1
                    else:
                        continue  # no change if on left boundry or capacity already reached
                elif r < prob_move_right + prob_move_left: #cell moves right
                    if x < 200 and new_epi.get((x+1, y, z), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x+1, y, z)] = new_epi[(x+1, y, z)] + 1 if (x+1, y) in new_epi else 1
                    else:
                        continue  # no change if on right boundry or capacity already reached
                elif r < prob_move_down + prob_move_right + prob_move_left:  # cell moves down
                    if y > 0 and new_epi.get((x, y-1, z), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x, y-1, z)] = new_epi[(x, y-1, z)] + 1 if (x, y-1, z) in new_epi else 1
                    else:
                        continue  # no change if on lower boundry or capacity already reached
                elif r < prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves up
                    if y < 200 and new_epi.get((x, y+1, z), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x, y+1, z)] = new_epi[(x, y+1, z)] + 1 if (x, y+1, z) in new_epi else 1
                    else:
                        continue  # no change if on upper boundry or capacity already reached
                    
                elif r < prob_move_back + prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves back
                    if z > 0 and new_epi.get((x, y, z-1), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x, y, z-1)] = new_epi[(x, y, z-1)] + 1 if (x, y, z-1) in new_epi else 1
                    else:
                        continue  # no change if on back boundry or capacity already reached
                elif r < prob_move_front + prob_move_back + prob_move_down + prob_move_right + prob_move_left + prob_move_up: # cell moves front
                    if z < height - 1 and new_epi.get((x, y, z+1), 0) <= 3:
                        new_epi[(x, y, z)] = new_epi[(x, y, z)] - 1
                        new_epi[(x, y, z+1)] = new_epi[(x, y, z+1)] + 1 if (x, y, z+1) in new_epi else 1
                    else:
                        continue  # no change if on front boundry or capacity already reached
        self.epi = new_epi
        
    def update_epithelial_mitosis(self) -> None:
        self.time+=dt
        if self.time % T_E == 0: 
            new_epi = (self.epi).copy()
            for (x, y, z), concentration in self.epi.items():
                new_epi[(x, y, z)] = concentration * 2 if concentration <= 4 else 4
            self.epi = new_epi
        
    #Sarah    
    def update_all(self, clusters: List[tuple[int, int]]) -> None: # list of cells that move into vasculature
        """
        update MMP2 then ECM then mesechymal then epithelial
        figures out what cells are coming in through the blood vessels and add them to the dicts
        """
        self.update_MMP2()
        self.update_ECM()
        self.update_mesechymal_movement()
        self.update_epithelial_movement()
        self.update_mesechymal_mitosis()
        self.update_epithelial_mitosis()
        #need to do: figures out what cells are coming in through the blood vessels and add them to the dicts


"""
Set up vascular class 
- List of cell clusters in vascular (#mesenchyml, #epithelia, time)
"""
#Funny Joelle 😂😂😂
@dataclass
class Vascular:
    clusters: List[tuple[int, int, int]] = field(default_factory=list) # (# mes, # epi, time)
    bones: List[tuple[int, int]] = field(default_factory=list)
    lungs: List[tuple[int, int]] = field(default_factory=list)
    liver: List[tuple[int, int]] = field(default_factory=list)

    def probExit(self, time, k, midpoint):
        """
        returns a probability of a cell leaving the vasculature
        probability increases as time increases
        time = how long cell has been in vasculature
        k = curve smoothness; default = .5
        midpoint = 1/2 vasc_time. So when time == midpint, probability of exiting = .5
        """
        return 1 / (1 + math.exp(-k * (time - midpoint)))
    
    def update_all(self, primary) -> None:
        """
        add the new cells to the current clusters
        iterate through the clusters
        increment time
        assign leaving clusters to new locations, and store in class attribute
        """
        newClusters = primary.clusters
        self.clusters += newClusters
        updatedClusters = []
        leavingVascular = []
        for cluster in self.clusters:
            mes = cluster[0]
            epi = cluster[1]
            time = cluster[2]
            leaveProb = self.probExit(time, .5, vasc_time/2) # added random time of leaving instead of fixed time
            if random.random() < leaveProb:
                leavingVascular.append(cluster)
                continue
            time +=1 #maybe this should be dt? -mia
            # checking to see if they disaggregate
            if time >= vasc_time/2 and mes+epi >1:
                disaggregate_mes = 0
                for _ in range(mes):
                    r = random.random()
                    if r < P_d:
                        disaggregate_mes += 1
                disaggregate_epi = 0
                for _ in range(epi):
                    r = random.random()
                    if r < P_d:
                        disaggregate_epi += 1
                remaining_mes = mes - disaggregate_mes
                remaining_epi = epi - disaggregate_epi
                if remaining_mes + remaining_epi >=2:
                    updatedClusters.append((remaining_mes, remaining_epi, time))
                for i in range(disaggregate_mes):
                    updatedClusters.append((1, 0, time))
                for i in range(disaggregate_epi):
                    updatedClusters.append((0, 1, time))
            else:
                updatedClusters.append((mes, epi, time))
        bones = []
        lungs = []
        liver = []
        for cluster in leavingVascular:
            if cluster[0] == 0 or cluster[1] == 0:
                prob = P_s # it's a single
            else:
                prob = P_C # it's a cluster
            r = random.random()
            if r < prob:
                continue
            else:
                newLoc = random.randrange(1, 4)
                if newLoc == 1:
                    bones.append((cluster[0], cluster[1]))
                elif newLoc == 2:
                    lungs.append((cluster[0], cluster[1]))
                elif newLoc == 3:
                    liver.append((cluster[0], cluster[1]))

        self.bones = bones
        self.lungs = lungs
        self.liver = liver

@dataclass
class Model: 
    """
    One primary breast grid object : (primary grid class)
    One vascular object : (vascular class)
    One secondy bones grid object : (secondary grid class)
    One secondary lungs grid object : (secondary grid class)
    One secondary liver grid object : (secondary grid class)
    move time step method?
    """
    breast: PrimaryGrid = field(default_factory=PrimaryGrid)
    vascular: Vascular = field(default_factory=Vascular)
    bones : SecondaryGrid = field(default_factory=SecondaryGrid)
    lungs : SecondaryGrid = field(default_factory=SecondaryGrid)
    liver : SecondaryGrid = field(default_factory=SecondaryGrid)
    
    #Sarah
    def initialize(self) -> None: 
        """
        initalize breast, vascular, bones, lungs, livers 
        populations breast with blood vessels, and cancer cells         
        """
        # Init breast tumor + vessels
        self.breast.initialize_primary()
        for g in [self.bones, self.lungs, self.liver]:
            g.initialize_secondary()
    
    #Baby
    def update(self) -> None: 
        """
        updates breast, vascular, bones, lungs, liver 
        """
        prev_primary_grid = self.breast
        self.breast.update_all()
        prev_vasc = self.vascular
        self.vascular.update_all(prev_primary_grid)
        # passing in the previous cells that will migrate to bones, lungs, liver
        self.bones.update_all(prev_vasc.bones)
        self.lungs.update_all(prev_vasc.lungs)
        self.liver.update_all(prev_vasc.liver)

    def preview(self, ax):
        """
        Create a visual of the grid
        """
        self.breast.preview(ax)

    
def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # create a model
    model = Model()
    # initialize model (populating with blood vessels and cells)
    model.initialize()
    # start with primary grid
    count = int(ITERATIONS / dt)
    # for t in range(count): change time later!!! # I think this shoudl be iterations / dt? # iterations
    def animate(i):
        # print(F"Updating model... iteration {t} / {count}")
        #update the model 
        model.update()
        
        model.preview(ax)
    _ = animation.FuncAnimation(fig, animate, interval=10)
    plt.show()

if __name__ == "__main__":
    main()
