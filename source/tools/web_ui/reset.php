<?php
    session_start();
    session_destroy();

	$output_dir = getcwd();
	exec("rm ". $output_dir ."/result.csv");
	exec("rm ". $output_dir ."/view_reference_sequence.csv");
	exec("rm ". $output_dir ."/view_sample_sequences.csv");
	exec("rm ". $output_dir ."/view_genotype_sequence.csv");
	exec("rm ". $output_dir ."/current_parameters.csv");

    header("Location: index.php");
?>