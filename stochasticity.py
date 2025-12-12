from collections import defaultdict
from dataclasses import dataclass, field
from typing import List
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import colors
from scipy.ndimage import convolve
import time
import math

WIDTH = 201  # board width
HEIGHT = 201  # board height

dt = 1 * (10 ** -3)
dx = 5 * (10 ** -3)
dy = 5 * (10 ** -3)
# Mesenchymal-like cancer cell diffusion coeff
#D_M = 1 * (10 ** -4)
D_M = 0.005
# Epithelial-like cancer cell diffusion coeff
#D_E = 5 * (10 ** -5)
D_E = 0.0025
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
#T_E = w
T_E = 0.1
# mesenchymal doubling time
#T_M = e
T_M = 0.15
# single CTC survival probability
P_s = 5 * (10 ** -4)
#P_s = 0
# cluster survival probability
P_C = 2.5 * (10 ** -4)
#P_C = 0
# extravasion prob to bones
E_1 = 0.5461
# extravasion prob to lungs
E_2 = 0.2553
# extravasion prob to liver
E_3 = 0.1986
# Number of iterations to run
ITERATIONS = 10
# Max steps in vasaclature
vasc_time = 0.18
# Probability cluster disaggregates
P_d = .3  # idk tbh the paper doesn't say...

# Global timing dictionary
timing_stats = defaultdict(list)

def time_function(func_name):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            timing_stats[func_name].append(elapsed)
            return result
        return wrapper
    return decorator

def print_timing_report():
    """Print a comprehensive timing report"""
    print("\n" + "=" * 80)
    print("PERFORMANCE PROFILING REPORT")
    print("=" * 80)
    
    # Calculate statistics for each function
    stats = []
    total_time = 0
    
    for func_name, times in timing_stats.items():
        if len(times) > 0:
            total = sum(times)
            avg = total / len(times)
            min_time = min(times)
            max_time = max(times)
            count = len(times)
            
            stats.append({
                'name': func_name,
                'total': total,
                'avg': avg,
                'min': min_time,
                'max': max_time,
                'count': count
            })
            total_time += total
    
    # Sort by total time (descending)
    stats.sort(key=lambda x: x['total'], reverse=True)
    
    # Print header
    print(f"\n{'Function':<35} {'Total (s)':<12} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Calls':<10} {'% Time':<10}")
    print("-" * 110)
    
    # Print each function's stats
    for stat in stats:
        pct = (stat['total'] / total_time * 100) if total_time > 0 else 0
        print(f"{stat['name']:<35} {stat['total']:>11.4f} {stat['avg']*1000:>11.4f} "
              f"{stat['min']*1000:>11.4f} {stat['max']*1000:>11.4f} {stat['count']:>9} {pct:>9.1f}%")
    
    print("-" * 110)
    print(f"{'TOTAL':<35} {total_time:>11.4f} {'':>11} {'':>11} {'':>11} {'':>9} {100:>9.1f}%")
    print("\n" + "=" * 80)
    
    # Identify bottlenecks
    print("\nBOTTLENECK ANALYSIS:")
    print("-" * 80)
    if len(stats) > 0:
        print(f"\nTop 3 slowest operations:")
        for i, stat in enumerate(stats[:3], 1):
            pct = (stat['total'] / total_time * 100) if total_time > 0 else 0
            print(f"  {i}. {stat['name']}: {stat['total']:.4f}s ({pct:.1f}% of total time)")
    
    print("\n" + "=" * 80)

def in_range(data, pt):
    if len(pt) != data.ndim:
        return False

    return all(0 <= pt[i] < data.shape[i] for i in range(len(pt)))

LAPLACIAN_KERNEL = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=np.float64)

cmap_epi = colors.LinearSegmentedColormap.from_list('epi',['black','blue'],256)
cmap_mes = colors.LinearSegmentedColormap.from_list('mes',["black", "green"],256)
cmap_bv = colors.LinearSegmentedColormap.from_list('bv',["black", "red"],256)
cmap_mmp2 = colors.LinearSegmentedColormap.from_list('mmp2',["black", "purple"],256)

cmap_mes._init()
cmap_epi._init()
cmap_bv._init()
cmap_mmp2._init()

alphas = np.linspace(0.0, 1.0, cmap_mes.N+3)
# cmap_mes._lut[:,-1] = alphas
# cmap_epi._lut[:,-1] = alphas
cmap_bv._lut[:,-1] = alphas

# alpha_mmp2 = np.linspace(0, 0.6, cmap_mes.N+3)
# cmap_mmp2._lut[:,-1] = alpha_mmp2

@dataclass
class Grid:
    """
    Represents 201 x 201 primary and secondary grids
    -5 dictionaries for mesenchymal, epithelial, MM2, ECM concentration, blood vessels (with keeping track of normal or ruptured)
    -key = location (tuple)
    -value = concentration (int, how many of them there are)
    -Specifc PDES (methods)
    """
    mes: np.ndarray = field(default_factory=lambda: np.zeros((WIDTH, HEIGHT), dtype=np.float32)) # mesenchymal
    epi: np.ndarray = field(default_factory=lambda: np.zeros((WIDTH, HEIGHT), dtype=np.float32)) # epithelial
    MMP2: np.ndarray = field(default_factory=lambda: np.zeros((WIDTH, HEIGHT), dtype=np.float32)) # matrix metalloproteinase-2
    ECM: np.ndarray = field(default_factory=lambda: np.ones((WIDTH, HEIGHT), dtype=np.float32)) # extracellular matrix
    bv: np.ndarray = field(default_factory=lambda: np.zeros((WIDTH, HEIGHT), dtype=np.int8)) # blood vessels
    clusters: List[tuple] = field(default_factory=list)  # clusters leaving
    time : float = 0 
    mes_time : float = 0 
    epi_time : float = 0 
    iterations : int = 0

    def preview(self, fig, ax, celltype = "breast"):
        # fig.suptitle("Cancer Cell Model ({}) - {} iterations".format(celltype, self.iterations))
        # ax[0].clear()
        # ax[0].set_title("Mesenchymal-like cancer cells")
        # ax[0].imshow(self.mes, cmap=cmap_mes)
        # ax[0].imshow(self.bv, cmap=cmap_bv)

        # ax[1].clear()
        # ax[1].set_title("Epithelial-like cancer cells")
        # ax[1].imshow(self.epi, cmap=cmap_epi)
        # ax[1].imshow(self.bv, cmap=cmap_bv)
        # Save this iteration's arrays

        if self.iterations % 10 == 0:
            np.savez_compressed(F'arrays/iter-{celltype}-{self.iterations}.npz',
                                epi=self.epi, bv=self.bv,
                                mes=self.mes, MMP2=self.MMP2, ECM=self.ECM)


    def __init__(self, gridtype = "primary"):
        self.mes = np.zeros((WIDTH, HEIGHT), dtype=np.float32)
        self.epi = np.zeros((WIDTH, HEIGHT), dtype=np.float32)
        self.MMP2 = np.zeros((WIDTH, HEIGHT), dtype=np.float32)
        self.ECM = np.ones((WIDTH, HEIGHT), dtype=np.float32)
        self.bv = np.zeros((WIDTH, HEIGHT), dtype=np.int8)
        self.clusters = []

        if gridtype == "primary":
            self.initialize_primary()
        else:
            self.initialize_secondary()

    # Rishika
    @time_function("Grid.initialize_primary")
    def initialize_primary(self):
        """
        adds mesechmmal cancer cells cluster in middle
        add epi
        """
        # I made my boundry flux thing assuming zero based indicies - Mia
        # find distances for all 201 points from center
        dist_points = {}
        dist_list = []
        for i in range(2, 199):
            for j in range(2, 199):
                distance = (i-100)**2+(j-100)**2
                dist_points[(i, j)] = distance
                dist_list.append(distance)
        dist_list.sort()
        # sort and use everything outside of first 200 in [2, 198] for 10 blood vessels
        twohundreth = dist_list[199]
        ninetyseventh = dist_list[96]
        min_coord, max_coord = 2, 198
        i = 0
        while i < 10:
            random_x = random.randint(min_coord, max_coord)
            random_y = random.randint(min_coord, max_coord)
            if (in_range(self.bv, (random_x, random_y)) and self.bv[random_x][random_y] != 0) or dist_points[(random_x, random_y)]<twohundreth:
                continue
            if i < 2:
                # Ruptured vessel
                self.bv[random_x][random_y] = 2
                self.bv[random_x-1][random_y] = 2
                self.bv[random_x][random_y-1] = 2
                self.bv[random_x][random_y+1] = 2
                self.bv[random_x+1][random_y] = 2
            else:
                self.bv[random_x][random_y] = 1
            i += 1
        
        # use middle 97 for the 388 cancer cells 
        # Cancer cells - center region
        y, x = np.ogrid[:WIDTH, :HEIGHT]
        dist_sq = (x - 100)**2 + (y - 100)**2
        center_mask = (dist_sq <= np.sort(dist_sq.ravel())[96]) & (x >= 2) & (x <= 198) & (y >= 2) & (y <= 198)
        center_coords = np.argwhere(center_mask)
        
        # Place 388 cells (155 epithelial, 233 mesenchymal)
        cell_indices = np.random.choice(len(center_coords), size=388, replace=True)
        for idx_num, idx in enumerate(cell_indices):
            i, j = center_coords[idx]
            if idx_num < 155:
                if self.epi[i, j] < 4:
                    self.epi[i, j] += 1
            else:
                if self.mes[i, j] + self.epi[i, j] < 4:
                    self.mes[i, j] += 1
    
    # Rishika
    @time_function("Grid.initialize_secondary")
    def initialize_secondary(self):
        """
        adds normal vessels to grid
        """
        min_coord, max_coord = 2, 198
        i = 0
        while i < 10:
            random_x = random.randint(min_coord, max_coord)
            random_y = random.randint(min_coord, max_coord)
            if (in_range(self.bv, (random_x, random_y)) and self.bv[random_x][random_y] != 0):
                continue
            self.bv[random_x][random_y] = 1
            i += 1

    # Carmen
    @time_function("Grid.update_MMP2")
    def update_MMP2(self) -> None:
        #m_new = m + dt*(Dm*laplacian(m) + theta*cm - lambda*m)
        # **NOTE: May have to move it back to normal later but rn it is not feasible to 
        # run simulation in a cost-effective manner without using a library for laplacian**

        laplacian_m = convolve(self.MMP2, LAPLACIAN_KERNEL, mode='nearest') / (dx**2)
        dm_dt = D_m * laplacian_m + theta * self.mes - MMP2_decay * self.MMP2
        self.MMP2 = np.maximum(0, self.MMP2 + dt * dm_dt)

    # Carmen
    @time_function("Grid.update_ECM")
    def update_ECM(self) -> None:
        #new_w = w + dt*-(gamma1*cm + gamma2*m)*w
        dw_dt = -(gamma_1 * self.mes + gamma_2 * self.MMP2) * self.ECM
        self.ECM = np.maximum(0, self.ECM + dt * dw_dt)

    # Mia
    @time_function("Grid.update_mesechymal_movement")
    def update_mesechymal_movement(self) -> None:
        """
        simulate movement
        simulate mitosis
        """
        new_mes = (self.mes).copy()
        rows, cols = self.mes.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                concentration = self.mes[i][j]
                for _ in range(0, int(concentration)):
                    ECM_conc_left = self.ECM[(i-1,j)]
                    ECM_conc_right = self.ECM[(i+1,j)]
                    ECM_conc_down = self.ECM[(i,j-1)]
                    ECM_conc_up = self.ECM[(i,j+1)]
                    z = random.random()
                    coeff = dt / (dx ** 2)
                    dw_dx = (ECM_conc_right - ECM_conc_left) / 2
                    dw_dy = (ECM_conc_up - ECM_conc_down) / 2

                    prob_left = coeff * (D_M - phi_M * dw_dx)
                    prob_right = coeff * (D_M + phi_M * dw_dx)
                    prob_down = coeff * (D_M - phi_M * dw_dy)
                    prob_up = coeff * (D_M + phi_M * dw_dy)

                    probs = np.maximum(np.array([prob_left, prob_right, prob_down, prob_up]), 0.0)
                    total = np.sum(probs)

                    [prob_move_left, prob_move_right, prob_move_down, prob_move_up] = probs / max(total, 1.0)

                    if z < prob_move_left:  # cell moves left
                        if i > 0 and new_mes[i-1][j] <= 3:
                            new_mes[(i, j)] = new_mes[(i, j)] - 1
                            new_mes[(i-1, j)] = new_mes[(i-1, j)] + 1
                        else:
                            continue  # no change if on left boundry or capacity already reached
                    elif z < prob_move_left + prob_move_right:  # cell moves right
                        if i < 200 and new_mes[i+1][j] <= 3:
                            new_mes[(i, j)] = new_mes[(i, j)] - 1
                            new_mes[(i+1, j)] = new_mes[(i+1, j)] + 1
                        else:
                            continue  # no change if on right boundry or capacity already reached
                    elif z < prob_move_down + prob_move_right + prob_move_left:  # cell moves down
                        if j > 0 and new_mes[i][j-1] <= 3:
                            new_mes[(i, j)] = new_mes[(i, j)] - 1
                            new_mes[(i, j-1)] = new_mes[(i, j-1)] + 1
                        else:
                            continue  # no change if on lower boundry or capacity already reached
                    elif z < prob_move_down + prob_move_right + prob_move_left + prob_move_up:
                        if j < 200 and new_mes[i][j+1] <= 3:
                            new_mes[(i, j)] = new_mes[(i, j)] - 1
                            new_mes[(i, j+1)] = new_mes[(i, j+1)] + 1
                        else:
                            continue  # no change if on upper boundry or capacity already reached
        self.mes = new_mes

    def update_mesechymal_mitosis(self) -> None:
        rows, cols = self.mes.shape
        for i in range(0, rows):
            for j in range(0, cols):
                interval = 1/ (T_M / dt)
                mitosisProb = 1 - math.exp(-interval)
                if random.random() < mitosisProb:
                    self.mes[i,j] = np.minimum(4, self.mes[i,j] * 2)


    # Mia
    @time_function("Grid.update_epithelial_movement")
    def update_epithelial_movement(self) -> None:
        """
        Simulate epithelial movement
        """
        new_epi = (self.epi).copy()
        rows, cols = self.epi.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                cell_count = int(self.epi[i][j])
                for _ in range(0, cell_count):
                    ECM_conc_left = self.ECM[(i-1,j)]
                    ECM_conc_right = self.ECM[(i+1,j)]
                    ECM_conc_down = self.ECM[(i,j-1)]
                    ECM_conc_up = self.ECM[(i,j+1)]
                    dw_dx = (ECM_conc_right - ECM_conc_left) / 2
                    dw_dy = (ECM_conc_up - ECM_conc_down) / 2
                    coeff = dt / (dx ** 2)

                    prob_left = coeff * (D_E - phi_E * dw_dx)
                    prob_right = coeff * (D_E + phi_E * dw_dx)
                    prob_down = coeff * (D_E - phi_E * dw_dy)
                    prob_up = coeff * (D_E + phi_E * dw_dy)

                    probs = np.maximum(np.array([prob_left, prob_right, prob_down, prob_up]), 0.0)
                    total = np.sum(probs)
                    if total > 1:
                        probs /= total
                    [prob_move_left, prob_move_right, prob_move_down, prob_move_up] = probs

                    z = random.random()
                    if z < prob_move_left:
                        if i > 0 and new_epi[i-1][j] <= 3:
                            new_epi[(i, j)] = new_epi[(i, j)] - 1
                            new_epi[(i-1, j)] = new_epi[(i-1, j)] + 1
                        else:
                            continue
                    elif z < prob_move_right + prob_move_left:
                        if i < 200 and new_epi[i+1][j] <= 3:
                            new_epi[(i, j)] = new_epi[(i, j)] - 1
                            new_epi[(i+1, j)] = new_epi[(i+1, j)] + 1
                        else:
                            continue
                    elif z < prob_move_down + prob_move_right + prob_move_left:
                        if j > 0 and new_epi[i][j-1] <= 3:
                            new_epi[(i, j)] = new_epi[(i, j)] - 1
                            new_epi[(i, j-1)] = new_epi[(i, j-1)] + 1
                        else:
                            continue
                    elif z < prob_move_down + prob_move_right + prob_move_left + prob_move_up:
                        if j < 200 and new_epi[i][j+1] <= 3:
                            new_epi[(i, j)] = new_epi[(i, j)] - 1
                            new_epi[(i, j+1)] = new_epi[(i, j+1)] + 1
                        else:
                            continue
        self.epi = new_epi

    def update_epithelial_mitosis(self) -> None:
        rows, cols = self.epi.shape
        for i in range(0, rows ):
            for j in range(0, cols):
                interval = 1/ (T_E / dt)
                mitosisProb = 1 - math.exp(-interval)
                if random.random() < mitosisProb:
                    self.epi[i,j] = np.minimum(4, self.epi[i,j] * 2)
    
    @time_function("Grid.extravasate_clusters")
    def extravasate_clusters(self, incoming_clusters, prob):
        """
        Upon extravasion, tumor cells exit vasculature into
        secondary tissue via vessel node, then disperse locally

        Ex. for each cluster going to grid:
        - cluster = (mes, epi)
        - Select random vessel coordinate (i, j)
        - Place cells into (i, j) or neighboring sites until all cells assigned

        Rules:
        - Max occupancy = 4
        - If cell can fit at (i, j), place into neighbor
        - If all neighbors full, drop excess cells
        """

        if np.random.random() > prob:
            # Can't extravasate, likelihood wasn't good enough
            return
        

        if not incoming_clusters:
            return
        # Blood vessel locations
        vessel_coords = np.argwhere(self.bv > 0)
        if len(vessel_coords) == 0:
            # No bvs to extravasate to
            return
        
        for mes_count, epi_count in incoming_clusters:
            # Choose a random one of the vessels
            idx = random.randint(0, len(vessel_coords) - 1)
            i, j = vessel_coords[idx]

            # Take all coords around an incoming cluster
            queue = [
                (x, y) for (x, y) in
                [(i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                if 0 <= x < WIDTH and 0 <= y < HEIGHT
            ]

            remaining_m = mes_count
            remaining_e = epi_count

            for (x, y) in queue:
                # Move mes cells into this new space around i, j if
                # there's room (if space > 0)
                space = 4 - (self.mes[x, y] + self.epi[x, y])
                if space <= 0:
                    continue

                
                add = min(space, remaining_m)
                self.mes[x, y] += add
                remaining_m -= add
                if remaining_m == 0:
                    break
            
            for (x, y) in queue:
                # Move epi cells into this new space around i, j if
                # there's room (if space > 0)
                space = 4 - (self.mes[x, y] + self.epi[x, y])
                if space <= 0:
                    continue
                
                add = min(space, remaining_e)
                self.epi[x, y] += add
                remaining_e -= add
                if remaining_e == 0:
                    break
            


    @time_function("Grid.find_intravasating_clusters")
    def find_intravasating_clusters(self):
        # List: (mes count, epi count)
        clusters = []

        mes = self.mes
        epi = self.epi

        def neighbors4(i, j):
            out = []
            if i > 0: out.append((i - 1, j))
            if i < WIDTH - 1: out.append((i + 1, j))
            if j > 0: out.append((i, j - 1))
            if j < WIDTH - 1: out.append((i, j + 1))
            return out

        # Set of visited clusters
        visited = set()

        rows, cols = self.epi.shape
        for i in range(rows):
            for j in range(cols):
                vessel_type = self.bv[i][j]
                # No vessel there, skip
                if vessel_type not in (1, 2):
                    continue

                num_mes = mes[i][j]
                num_epi = epi[i][j]
                to_collect = []

                if vessel_type == 1:
                    if num_mes == 0:
                        continue
                
                    to_collect = [(i, j), *neighbors4(i, j)]
                elif vessel_type == 2:
                    if num_mes == 0 and num_epi == 0:
                        continue
                
                    to_collect = [(i, j)] + neighbors4(i, j)
                
                total_mes = 0
                total_epi = 0

                for loc in to_collect:
                    # Location's been visited already, skip
                    if loc in visited:
                        continue
                    visited.add(loc)
                    i, j = loc

                    total_mes += mes[i][j]
                    total_epi += epi[i][j]
                
                for loc in to_collect:
                    if in_range(mes, loc):
                        i, j = loc
                        mes[i][j] = 0
                    if in_range(epi, loc):
                        i, j = loc
                        epi[i][j] = 0
                
                if total_mes + total_epi > 0:
                    clusters.append((total_mes, total_epi, 0))
            
        return clusters

    # Sarah
    def update_all(self, primary = None) -> None:  # list of cells that move into vasculature
        """
        update MMP2 then ECM then mesechymal then epithelial
        figures out what cells are on blood vessels and removes them from dicts
        adds clusters leaving to self.cluster
        """
        if primary is None:
            self.update_MMP2()
            self.update_ECM()
            self.update_mesechymal_movement()
            self.update_epithelial_movement()

            new_clusters = self.find_intravasating_clusters()

            self.clusters.extend(new_clusters)
            self.update_mesechymal_mitosis()
            self.update_epithelial_mitosis()
        else:
            """
            add the new cells to the current clusters
            iterate through the clusters
            increment time
            assign leaving clusters to new locations, and store in class attribute
            """
            self.update_MMP2()
            self.update_ECM()
            self.update_mesechymal_movement()
            self.update_epithelial_movement()

            new_clusters = self.find_intravasating_clusters()

            self.clusters.extend(new_clusters)
            self.update_mesechymal_mitosis()
            self.update_epithelial_mitosis()
        
        self.iterations += 1



"""
Set up vascular class 
- List of cell clusters in vascular (#mesenchyml, #epithelia, time)
"""
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
            leaveProb = self.probExit(time, .5, vasc_time / 2)  # added random time of leaving instead of fixed time
            if random.random() < leaveProb:
                leavingVascular.append(cluster)
                continue
            time += dt #maybe this should be dt? -mia 
            # checking to see if they disaggregate
            if time >= vasc_time/2 and mes+epi >1:
                disaggregate_mes = 0
                for _ in range(int(mes)):
                    r = random.random()
                    if r < P_d:
                        disaggregate_mes += 1
                disaggregate_epi = 0
                for _ in range(int(epi)):
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
        self.clusters = updatedClusters
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
    breast: Grid = field(default_factory=Grid)
    vascular: Vascular = field(default_factory=Vascular)
    bones : Grid = field(default_factory=lambda: Grid("secondary"))
    lungs : Grid = field(default_factory=lambda: Grid("secondary"))
    liver : Grid = field(default_factory=lambda: Grid("secondary"))
    
    #Sarah
    def initialize(self) -> None: 
        """
        initalize breast, vascular, bones, lungs, livers 
        populations breast with blood vessels, and cancer cells         
        """
        # Init breast tumor + vessels
        print("Initializing")
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
        
        # Extravasion:
        self.bones.extravasate_clusters(prev_vasc.bones, E_1)
        self.lungs.extravasate_clusters(prev_vasc.lungs, E_2)
        self.liver.extravasate_clusters(prev_vasc.liver, E_3)

        # passing in the previous cells that will migrate to bones, lungs, liver
        self.bones.update_all(prev_vasc.bones)
        self.lungs.update_all(prev_vasc.lungs)
        self.liver.update_all(prev_vasc.liver)

    def preview(self, fig, ax):
        """
        Create a visual of the grid
        """
        self.breast.preview(fig, ax, "breast")
        self.bones.preview(fig, ax, "bones")
        self.lungs.preview(fig, ax, "lungs")
        self.liver.preview(fig, ax, "liver")

    
def main():
    # fig = plt.figure()
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs = np.array(axs)  # ensures indexing works even if ncols=1
    # create a model
    model = Model()
    # initialize model (populating with blood vessels and cells)
    model.initialize()
    # start with primary grid
    count = int(ITERATIONS / dt)
    # for t in range(count): change time later!!! # I think this shoudl be iterations / dt? # iterations
    def run_simulation(model, steps):
        frames = {
            "breast":  {"mes": [], "epi": [], "mmp": [], "ecm": [], "bv": []},
            "lungs":   {"mes": [], "epi": [], "mmp": [], "ecm": [], "bv": []},
            "bones":   {"mes": [], "epi": [], "mmp": [], "ecm": [], "bv": []},
            "liver":   {"mes": [], "epi": [], "mmp": [], "ecm": [], "bv": []}
        }

        for i in range(steps):
            model.update()
            if i % 10 == 0:
                print(i)
                for organ_name, organ in [
                    ("breast", model.breast),
                    ("lungs",  model.lungs),
                    ("bones",  model.bones),
                    ("liver",  model.bones)
                ]:
                    frames[organ_name]["mes"].append(organ.mes.copy())
                    frames[organ_name]["epi"].append(organ.epi.copy())
                    frames[organ_name]["mmp"].append(organ.MMP2.copy())
                    frames[organ_name]["ecm"].append(organ.ECM.copy())
                    frames[organ_name]["bv"].append(organ.bv.copy())

        return frames
    
    def gif_from_frames(frames, filename, cmap="viridis"):

        fig, axs = plt.subplots(1, 5, figsize=(16, 5))

        def animate(i):
            axs[0].clear(); axs[0].imshow(frames["mes"][i]); axs[0].set_title("MES"); axs[0].set_axis_off()
            axs[1].clear(); axs[1].imshow(frames["epi"][i]); axs[1].set_title("EPI"); axs[1].set_axis_off()
            axs[2].clear(); axs[2].imshow(frames["mmp"][i]); axs[2].set_title("MMP2"); axs[2].set_axis_off()
            axs[3].clear(); axs[3].imshow(frames["ecm"][i]); axs[3].set_title("ECM"); axs[3].set_axis_off()
            axs[4].clear(); axs[4].imshow(frames["bv"][i]); axs[4].set_title("BV"); axs[4].set_axis_off()

        ani = animation.FuncAnimation(fig, animate, frames=len(frames["mes"]), interval=1)
        ani.save(filename, writer="pillow")
        plt.close(fig)

    frames = run_simulation(model, steps=2000)

    gif_from_frames(frames["breast"], "breast2.gif")
    gif_from_frames(frames["lungs"],  "lungs2.gif")
    gif_from_frames(frames["bones"],  "bones2.gif")
    gif_from_frames(frames["liver"],  "liver2.gif")

if __name__ == "__main__":
    main()