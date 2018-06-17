<?php
	session_start();
	error_reporting(E_ALL);

	$output_dir = getcwd();

	$parameter=$_REQUEST['parameter'];
	$value=$_REQUEST['value'];
	
	error_log($parameter);
	error_log($value);

	$query = "";
	if($parameter == "mimimum") {
		$query = "set genotype_frequency_min=" . $value;
	} else if ($parameter == "maximum") {
		$query = "set genotype_frequency_max=" . $value;
	}
	
	exec("echo \"".$query."\" | /bin/nc.openbsd localhost 8000");

	$query = "select * from sys_variables;";
	
	exec("echo \"".$query."\" | /bin/nc.openbsd localhost 8000 > "  . $output_dir  ."/current_parameters.csv");
?>