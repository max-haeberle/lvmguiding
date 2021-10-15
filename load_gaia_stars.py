from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import numpy as np

import astropy.units as u




def load_gaia_stars(c, search_radius,bright_limit=-99,faint_limit=99):

    inner_search_radius=0
    outer_search_radius=search_radius

    c_icrs=c.transform_to('icrs')   

    radius = u.Quantity(outer_search_radius, u.deg)

    query_string = "SELECT TOP 99999 source_id, ra,dec,phot_g_mean_mag FROM gaiaedr3.gaia_source WHERE phot_g_mean_mag <= "+str(faint_limit)+" AND 1=CONTAINS(POINT('ICRS',ra,dec), CIRCLE('ICRS',"+str(c_icrs.ra.deg)+","+str(c_icrs.dec.deg)+", "+str(radius.value)+"))"

    print("Gaia query: ",query_string)
    j = Gaia.launch_job(query_string,verbose=True)
    cat = j.get_results()
            #print("Gaia query: ",gaia_query)
    print(f'{len(cat)} stars found within {search_radius} deg')


        
    return cat









