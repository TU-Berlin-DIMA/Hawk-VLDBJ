<?php
	session_start();
	error_reporting(E_ALL);

	$output_dir = getcwd();

	$contigId=$_REQUEST['contigid'];
	$start=$_REQUEST['start'];
	$end=$_REQUEST['end'];

	$_SESSION['contigid_view'] = $contigId;
	$_SESSION['start_view'] = $start;
	$_SESSION['end_view'] = $end;

	// query for reference sequence
	$reference_sequence_query = "select concat_bases(rb_base_value), min(rb_position) as start, max(rb_position) as end from reference_base where rb_c_id = ". $contigId . " and rb_position between " . $start . " and " . $end . ";";

	exec("echo \"".$reference_sequence_query."\" | /bin/nc.openbsd localhost 8000 > ". $output_dir ."/view_reference_sequence.csv");

	// query for sample sequences
	$sample_sequences_query = "select sb_read_id, concat_bases(sb_base_value) as read, min(rb_position) as start, max(rb_position) as end from select sb_read_id, sb_base_value, rb_position	from sample_base join reference_base on sb_rb_id = rb_id where rb_c_id = " . $contigId . " and rb_position between " . $start . " and " . $end . " order by rb_position, sb_insert_offset group by sb_read_id order by sb_read_id;";

	exec("echo \"".$sample_sequences_query."\" | /bin/nc.openbsd localhost 8000 > ". $output_dir ."/view_sample_sequences.csv");

	// query for genotype sequence
	$sample_sequences_query = "select concat_bases(my_genotype) from select rb_position, rb_base_value, sb_insert_offset, genotype(sb_base_value) as my_genotype from sample_base join reference_base on sb_rb_id = rb_id where sb_insert_offset = 0 and rb_c_id = " . $contigId . " and rb_position between " . $start . " and " . $end . " group by rb_position, rb_base_value, sb_insert_offset order by rb_position, sb_insert_offset;";

	exec("echo \"".$sample_sequences_query."\" | /bin/nc.openbsd localhost 8000 > ". $output_dir ."/view_genotype_sequence.csv");
?>