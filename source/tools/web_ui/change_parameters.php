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
    <script type="text/javascript" src="js/codemirror.js"></script>
    <script type="text/javascript" src="js/codemirror_modes/sql.js"></script>

    <script type="text/javascript" src="js/change_parameters.js"></script>

</head>
<body>

<?php
    session_start();
    include ("include/navigation.php");
?>

<script type="text/javascript">

</script>

<div class="container">
    <div class="row">
        <div class="col-lg-4">
            <hr>
            <b>Change parameterization</b>
            <hr>
            <div class="form-group">
                <label for="selectMinimumFrequency">Minimum genotype frequency:</label>
                <select id="selectMinimumFrequency" class="form-control pull-left">
                    <option>0.1</option>
                    <option>0.1</option>
                    <option selected="true">0.2</option>
                    <option>0.3</option>
                    <option>0.4</option>
                    <option>0.5</option>
                    <option>0.6</option>
                    <option>0.7</option>
                    <option>0.8</option>
                    <option>0.9</option>
                    <option>1.0</option>
                </select> 
            </div>
            <div class="form-group">
                <label for="selectMaximumFrequency">Maximum genotype frequency:</label>
                <select id="selectMaximumFrequency" class="form-control pull-left">
                    <option>0.1</option>
                    <option>0.1</option>
                    <option>0.2</option>
                    <option>0.3</option>
                    <option>0.4</option>
                    <option>0.5</option>
                    <option>0.6</option>
                    <option>0.7</option>
                    <option selected="true">0.8</option>
                    <option>0.9</option>
                    <option>1.0</option>
                </select> 
            </div>  
        </div>                
        <div class="col-lg-8">
            <hr>
            <b>Current parameters</b><br>
            <hr>
            <?php
                if(($csv_file = fopen("current_parameters.csv", "r")) !== FALSE) {                    
            ?>
            <table id="parametersTable" class="table table-condensed" data-align="left" data-halign="left" data-pagination="true" data-search="false">
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
                        if(trim($line[0]) == "genotype_frequency_max" || trim($line[0]) == "genotype_frequency_min") {
                            foreach ($line as $cell) {?>
                                <td><?php echo trim(htmlspecialchars($cell)); ?></td>
                    <?php
                            }
                        }
                    }
                    ?>
                    </tr>
                </tbody>
            </table>
            <?php
                } else {
                    echo "Current parameterization not available.";
                }
                fclose($csv_file);
            ?>
        </div>
    </div>
</div>
</body>
</html>