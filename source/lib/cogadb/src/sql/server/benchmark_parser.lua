#!/usr/bin/lua

local ssb_queries = {
	--"select sum(lo_extendedprice*lo_discount) as revenue from lineorder, dates where lo_orderdate = d_datekey and d_year = 1993 and lo_discount between 1 and 3 and lo_quantity < 25;",
	{"1.1", "select sum(lo_extendedprice*lo_discount) as revenue from lineorder, dates where lo_orderdate = d_datekey and d_year = 1993 and lo_discount between 1.0 and 3.0 and lo_quantity < 25;"},
	--"select sum(lo_extendedprice*lo_discount) as revenue from lineorder, dates where lo_orderdate = d_datekey and d_yearmonthnum = 199401 and lo_discount between 4 and 6 and lo_quantity between 26 and 35;",
	{"1.2", "select sum(lo_extendedprice*lo_discount) as revenue from lineorder, dates where lo_orderdate = d_datekey and d_yearmonthnum = 199401 and lo_discount between 4.0 and 6.0 and lo_quantity between 26 and 35;"},
	--"select sum(lo_extendedprice*lo_discount) as revenue from lineorder, dates where lo_orderdate = d_datekey and d_weeknuminyear = 6 and d_year = 1994 and lo_discount between 5 and 7 and lo_quantity between 26 and 35;",
	{"1.3", "select sum(lo_extendedprice*lo_discount) as revenue from lineorder, dates where lo_orderdate = d_datekey and d_weeknuminyear = 6 and d_year = 1994 and lo_discount between 5.0 and 7.0 and lo_quantity between 26 and 35;"},

	{"2.1", "select sum(lo_revenue), d_year, p_brand from lineorder, dates, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA' group by d_year, p_brand order by d_year, p_brand;"},
	{"2.2", "select sum(lo_revenue), d_year, p_brand from lineorder, dates, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_brand between 'MFGR#2221' and 'MFGR#2228' and s_region = 'ASIA' group by d_year, p_brand order by d_year, p_brand;"},
	{"2.3", "select sum(lo_revenue), d_year, p_brand from lineorder, dates, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_brand= 'MFGR#2239' and s_region = 'EUROPE' group by d_year, p_brand order by d_year, p_brand;"},

	--"select c_nation, s_nation, d_year, sum(lo_revenue) as revenue from customer, lineorder, supplier, dates where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and c_region = 'ASIA' and s_region = 'ASIA' and d_year >= 1992 and d_year <= 1997 group by c_nation, s_nation, d_year order by d_year asc, revenue desc;"
	{"3.1", "select c_nation, s_nation, d_year, sum(lo_revenue) as revenue from customer, lineorder, supplier, dates where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and c_region = 'ASIA' and s_region = 'ASIA' and d_year >= 1992 and d_year <= 1997 group by c_nation, s_nation, d_year order by d_year asc, lo_revenue desc;"},
	--"select c_city, s_city, d_year, sum(lo_revenue) as revenue from customer, lineorder, supplier, dates where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and c_nation = 'UNITED STATES' and s_nation = 'UNITED STATES' and d_year >= 1992 and d_year <= 1997 group by c_city, s_city, d_year order by d_year asc, lo_revenue desc;"
	{"3.2", "select c_city, s_city, d_year, sum(lo_revenue) as revenue from customer, lineorder, supplier, dates where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and c_nation = 'UNITED STATES' and s_nation = 'UNITED STATES' and d_year >= 1992 and d_year <= 1997 group by c_city, s_city, d_year order by d_year asc, lo_revenue desc;"},
	--select c_city, s_city, d_year, sum(lo_revenue) as revenue from customer, lineorder, supplier, dates where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and (c_city='UNITED KI1' or c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5') and d_year >= 1992 and d_year <= 1997 group by c_city, s_city, d_year order by d_year asc, revenue desc;"
	{"3.3", "select c_city, s_city, d_year, sum(lo_revenue) as revenue from customer, lineorder, supplier, dates where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and (c_city='UNITED KI1' or c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5') and d_year >= 1992 and d_year <= 1997 group by c_city, s_city, d_year order by d_year asc, lo_revenue desc;"},
	--"select c_city, s_city, d_year, sum(lo_revenue) as revenue from customer, lineorder, supplier, dates where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and (c_city='UNITED KI1' or c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5') and d_yearmonth = 'Dec1997' group by c_city, s_city, d_year order by d_year asc, revenue desc;"
	{"3.4", "select c_city, s_city, d_year, sum(lo_revenue) as revenue from customer, lineorder, supplier, dates where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate = d_datekey and (c_city='UNITED KI1' or c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5') and d_yearmonth = 'Dec1997' group by c_city, s_city, d_year order by d_year asc, lo_revenue desc;"},

	{"4.1", "select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit from dates, customer, supplier, part, lineorder where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey and c_region = 'AMERICA' and s_region = 'AMERICA' and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') group by d_year, c_nation order by d_year, c_nation;", false},
	{"4.2", "select d_year, s_nation, p_category, sum(lo_revenue - lo_supplycost) as profit from dates, customer, supplier, part, lineorder where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey and c_region = 'AMERICA' and s_region = 'AMERICA' and (d_year = 1997 or d_year = 1998) and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') group by d_year, s_nation, p_category order by d_year, s_nation, p_category;", false},
	{"4.3", "select d_year, s_city, p_brand, sum(lo_revenue - lo_supplycost) as profit from dates, customer, supplier, part, lineorder where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey and s_nation = 'UNITED STATES' and (d_year = 1997 or d_year = 1998) and p_category = 'MFGR#14' group by d_year, s_city, p_brand order by d_year, s_city, p_brand;", false}
}

local measurements_per_query = 100
local truncate_measurements = 10

function writef(hnd, ...) hnd:write(string.format(...)) end

function benchmark_query(query, execute)
	local hnd = io.popen(string.format("(bin/sql_benchmark_parser %s %q >/dev/null) 2>&1",
					   execute and "" or "--no-execute", query))
	local parse_time, total_time

	for line in hnd:lines() do
		parse_time = tonumber(line:match("Parse: (%d+) ns")) or parse_time
		total_time = tonumber(line:match("Total: (%d+) ns")) or total_time
	end

	return parse_time, total_time
end

local grap = io.open("benchmark_parser.g", "w+")

local tbl = io.open("benchmark_parser.tbl", "w+")
tbl:write(".TS\n"..
	  "allbox,center,tab(;);\n"..
	  "LB LB LB LB\nN N N N.\n"..
	  "SSB;Parsezeit (in \\(mcs);Gesamtabarbeitungszeit (in ms);`Parsezeit over Gesamtabarbeitungszeit`\n")

for _, query_spec in ipairs(ssb_queries) do
	local query_no, query, execute = table.unpack(query_spec)
	if execute == nil then execute = true end

	local times = {}
	local avg_parse_time, avg_total_time = 0, 0

	print("Measuring query "..query_no)

	for i = 1, measurements_per_query do
		local parse_time, total_time = benchmark_query(query, execute)
		assert(type(parse_time) == "number")
		assert(type(total_time) == "number")

		writef(io.stdout, "Query %s: %d/100: %d/%d ns (%g%%)\n",
		       query_no, i, parse_time, total_time,
		       (parse_time*100)/total_time)
		table.insert(times, {parse = parse_time, total = total_time})
	end

	-- calculate truncated average
	-- discard 10 lowest and highest measurements
	table.sort(times, function(a, b)
		return (a.parse/a.total) < (b.parse/b.total)
	end)
	for i = 1+truncate_measurements, measurements_per_query-truncate_measurements do
		avg_parse_time = avg_parse_time + times[i].parse
		avg_total_time = avg_total_time + times[i].total
	end
	avg_parse_time = avg_parse_time/(measurements_per_query-2*truncate_measurements)
	avg_total_time = avg_total_time/(measurements_per_query-2*truncate_measurements)

	writef(grap, "%s %d %d\n", query_no, avg_parse_time, avg_total_time)
	writef(tbl, "%s;%s;%s;%s\n", query_no,
	       string.format("%.3g", avg_parse_time/1000),
	       avg_total_time > 0 and string.format("%.6g", avg_total_time/1000000) or "\\(em",
	       avg_total_time > 0 and string.format("%.2g%%", (avg_parse_time*100)/avg_total_time) or "\\(em")
end

tbl:write(".TE\n")
tbl:close()

grap:close()

