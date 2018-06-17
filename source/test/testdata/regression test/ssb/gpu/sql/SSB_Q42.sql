select d_year, s_nation, p_category, sum(lo_revenue - lo_supplycost) as profit from lineorder JOIN supplier ON (lo_suppkey = s_suppkey) JOIN dates ON (lo_orderdate = d_datekey) JOIN part ON (lo_partkey = p_partkey) JOIN customer ON (lo_custkey = c_custkey) where c_region = 'AMERICA' and s_region = 'AMERICA' and (d_year = 1997 or d_year = 1998) and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') group by d_year, s_nation, p_category order by d_year, s_nation, p_category;
