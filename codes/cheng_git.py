# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:35:01 2019

@author: ycai
"""
import xgboost as xgb
import pandas as pd
import numpy as np

def attack_untargeted(model, dtest, x0, y0, nclasses, alpha = 0.2, beta = 0.001, iterations = 1000):
    dataset = 'breast_cancer'
    model = xgb.Booster()
    model_path = '/home/cai7/models/xgb/{}_xgb.model'.format(dataset)
    model.load_model(model_path)
    test_df = pd.read_pickle('/home/cai7/chosen_sample/xgb/{}_xgb_samples.pkl'.format(dataset))

    test_data = np.array(test_df.drop(columns = ['label']))
    test_label = test_df['label'].tolist()
    dtest = xgb.DMatrix(test_data, label = test_label)
    
    best_theta, g_theta, dis = None, float('inf'), float('inf')
    for i in range(len(test_label)):
        if predict(model, test_data[i], nclasses) != y0:
            theta = test_data[i] - x0
            initial_lbd = 1.0            
            lbd, distance = fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
            if distance < dis:
                best_theta, g_theta, dis = theta, lbd, distance

    g1 = 1.0
    theta, g2 = best_theta, g_theta
    stopping = 0.01
    prev_obj = 100000
    for i in range(iterations):
        gradient = torch.zeros(theta.size())
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = torch.randn(theta.size()).type(torch.FloatTensor)
            u = u/torch.norm(u)
            ttt = theta+beta * u
            ttt = ttt/torch.norm(ttt)
            g1, count = fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2, tol=beta/500)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0/q * gradient

        if (i+1)%50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))
            if g2 > prev_obj-stopping:
                break
            prev_obj = g2

        min_theta = theta
        min_g2 = g2
    
        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta = new_theta/torch.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
            opt_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta 
                min_g2 = new_g2
            else:
                break

        if min_g2 >= g2:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = theta - alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                if new_g2 < g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    break

        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = theta.clone(), g2
        
        #print(alpha)
        if alpha < 1e-4:
            alpha = 1.0
            print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if (beta < 0.0005):
                break

    target = model.predict(x0 + g_theta*best_theta)
    return x0 + g_theta*best_theta

def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd
     
    if model.predict(x0+lbd*theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while model.predict(x0+lbd_hi*theta) == y0:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi > 20:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while model.predict(x0+lbd_lo*theta) != y0 :
            lbd_lo = lbd_lo*0.99
            nquery += 1

    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def fine_grained_binary_search(model, x0, y0, theta, initial_lbd, current_best):
    if initial_lbd > current_best: 
        if predict(model, x0+current_best*theta, nclasses) == y0:
            return float('inf')
        lbd = current_best
    else:
        lbd = initial_lbd
    lbd_hi = lbd
    lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        if predict(model, x0 + lbd_mid*theta, nclasses) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    t = theta * lbd_hi
    dis = np.abs(min(t, key=abs))
    return lbd_hi, dis
