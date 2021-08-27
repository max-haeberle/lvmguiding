from astroquery.gaia import Gaia
import numpy as np
from astroquery.utils.tap.core import TapPlus
from time import sleep
from glob import glob1
#from numpy.lib.recfunctions import stack_arrays
import healpy as hp
import time

import os

# connecting to Gaia Archive
gaia = TapPlus(url="http://gea.esac.esa.int/tap-server/tap")
#gaia.login_gui()
gaia.login(user='mhaberle', password='xxx')

hpx_query = 6

NSIDE = 2**hpx_query

nhpx = hp.nside2npix(NSIDE)#tap.number_of_healpixels(hpx_query)


hpx_query_res = 2**35*4**(12-hpx_query)#tap.gaia_hpx_factor(hpx_query)
list_of_border_source_ids = []



for i in np.arange(nhpx+1):
    list_of_border_source_ids.append(i*hpx_query_res)

    
query = """SELECT source_id, ra,dec,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag FROM gaiaedr3.gaia_source
WHERE source_id BETWEEN %d AND %d"""




QUERIES = []
for i,item in enumerate(list_of_border_source_ids):
    if i == len(list_of_border_source_ids)-1:
        continue
    QUERIES.append(query %(list_of_border_source_ids[i],list_of_border_source_ids[i+1]-1))
    
    
    
t00 = time.time()

os.system("mkdir Gaia_Healpix_{:d}".format(hpx_query))

for i in range(11654,nhpx):
    try:
        t0 = time.time()
        print(i,nhpx)
        job = gaia.launch_job_async(QUERIES[i])
        print(i,job)
        r = job.get_results()
        print(len(r))
        np.save('./Gaia_Healpix_{:d}/lvl{:d}_{:06d}.npy'.format(hpx_query,hpx_query,i),r)
        lst = gaia.list_async_jobs(verbose=True)
        for i2 in lst:
                gaia.remove_jobs(i2.jobid)
        t1 = time.time()
        print("########################################################################################################################")
        print("Finished query {:6d} of {:6d}. Duration: {:.2f} s    Total passed time: {:.1f} s Estimate for completion: {:.1f}s ".format(i,nhpx,t1-t0,t1-t00,(t1-t00)/(i+1)*(nhpx-i)))
        #print("Finished query {:6d} of {:6d}. Duration: {:.2f} s    Total passed time: {:.1f} s ".format(i,nhpx,t1-t0,t1-t00))
        print("########################################################################################################################")
    except:
        print("reconnecting in 10sec")
        sleep(10)
        gaia = TapPlus(url="http://gea.esac.esa.int/tap-server/tap")
        gaia.login(user='mhaberle', password='xxx')
        print(i,nhpx)
        job = gaia.launch_job_async(QUERIES[i])
        print(i,job)
        r = job.get_results()
        print(len(r))
        np.save('./Gaia_Healpix_{:d}/lvl{:d}_{:06d}.npy'.format(hpx_query,hpx_query,i),r)
        lst = gaia.list_async_jobs(verbose=True)
        for i2 in lst:
                gaia.remove_jobs(i2.jobid)
                
                
        
