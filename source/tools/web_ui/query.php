<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>CoGaDB Web UI</title>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <meta name="viewport" content="width=device, initial-scale=1">
    <link rel="stylesheet" type="text/css" href="css/bootstrap.min.css">
    <link rel="stylesheet" type="text/css" href="css/bootstrap-table.min.css">
    <link rel="stylesheet" type="text/css" href="css/codemirror.css">

    <script type="text/javascript" src="js/jquery.min.js"></script>
    <script type="text/javascript" src="js/bootstrap.min.js"></script>
    <script type="text/javascript" src="js/bootstrap-table.min.js"></script>
    <script type="text/javascript" src="js/query.js"></script>
    <script type="text/javascript" src="js/codemirror.js"></script>
    <script type="text/javascript" src="js/codemirror_modes/sql.js"></script>

</head>
<body>

<?php
    session_start();
    include ("include/navigation.php");
?>

<script type="text/javascript">

// setup index to retrieve SQL query behind description
var available_queries = {
    <?php 
        if(($csv_file = fopen("available_queries.csv", "r")) !== FALSE) { 
            while (($line = fgetcsv($csv_file, 0, "|")) !== false) {
                echo "\"".$line[0] . "\":\"" . $line[1] . "\",\n";
            }
        }
    ?>  
}

var lastQuery = <?php echo "\"".$_SESSION['query']."\""?>;

</script>

<div class="container">
    <div class="row">
        <div class="col-lg-4">
            <form>
                <b>Enter your Query</b><br>
                <hr>
                <textarea style="width:100%;" class="form-control" id="textareaQuery" placeholder="Enter your query" rows="10"></textarea>            
            </form>
            <hr>
            <form>
              <select id="selectPredefinedQuery" class="form-control pull-left">
                    <option selected="true" style="display:none;"><?php echo (isset($_SESSION['query_name']) ? $_SESSION['query_name'] : 'Select a predefined query ...')?></option>
                    <?php 
                        if(($csv_file = fopen("available_queries.csv", "r")) !== FALSE) { 
                            while (($line = fgetcsv($csv_file, 0, "|")) !== false) {?>
                                <option>
                                    <?php echo htmlspecialchars($line[0]) ?>
                                </option>
                            <?php
                            }
                        }
                    ?>  
                </select> 
                <br>
            <hr>
                <button id="submitQuery" type="submit" class="btn btn-primary pull-right">Submit</button>
            </form>           
        </div>                
        <div class="col-lg-8">
            <b>Query result</b> - <?php printf("Execution Time: %.3f seconds", $_SESSION['runtime']); ?><br>
            <hr>
            <?php
                if(($csv_file = fopen("result.csv", "r")) !== FALSE) {                    
            ?>
            <table id="queryResultTable" class="table-condensed" data-align="left" data-halign="left" data-pagination="true" data-search="false" data-toggle="table">
                <thead>
                    <tr>
                    <?php                         
                        if(($line = fgetcsv($csv_file, 0, "|")) !== false) {
                            foreach ($line as $cell) { ?>
                                <th data-field="<?php echo trim(htmlspecialchars($cell)); ?>"><?php echo trim(htmlspecialchars($cell)); ?></th>            
                    <?php
                            }
                        }
                    ?>
                    </tr>
                </thead>
                <tbody>
                <?php
                    while (($line = fgetcsv($csv_file, 0, "|")) !== false) {?>
                    <tr>
                    <?php
                        foreach ($line as $cell) {?>
                            <td><?php echo trim(htmlspecialchars($cell)); ?></td>
                    <?php
                        }
                    }
                    ?>
                    </tr>
                </tbody>
            </table>
            <?php
                } else {
                    echo "Enter your query.";
                }
                fclose($csv_file);
            ?>
        </div>
    </div>
</div>
</body>
</html>