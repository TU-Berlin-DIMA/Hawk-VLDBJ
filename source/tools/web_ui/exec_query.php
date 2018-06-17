<?php
	session_start();
	error_reporting(E_ALL);

	$output_dir = getcwd();

	function microtime_float()
	{
    	list($usec, $sec) = explode(" ", microtime());
    	return ((float)$usec + (float)$sec);
	}

	$query=$_REQUEST['query'];
	$_SESSION['query'] = $query;
	$_SESSION['query_name'] = $_REQUEST['query_name'];

	$query = str_replace(":APOSTROPHE:", "'", $query);	
	$query = str_replace(":NEWLINE:", " ", $query);
	//$query = strtolower($query);
	//error_log('Query: '.$query);


	$time_start = microtime_float();

	exec("echo \"".$query."\" | /bin/nc.openbsd localhost 8000 > ". $output_dir ."/result.csv");

	$time_end = microtime_float();

	$time = $time_end - $time_start;

	$_SESSION['runtime'] = $time;
?>