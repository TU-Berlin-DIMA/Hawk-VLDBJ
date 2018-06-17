<?php
	session_start();
	error_reporting(E_ALL);

	$output_dir = getcwd();
	
	$query = "select * from sys_variables;";
	
	exec("echo \"".$query."\" | /bin/nc.openbsd localhost 8000 > "  . $output_dir  ."/current_parameters.csv");
?>