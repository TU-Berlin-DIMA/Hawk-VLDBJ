$(function() {

    $('#parametersTable').bootstrapTable('hideLoading');

    $('#selectMinimumFrequency').on('click', function () {
        var selectedValue = $('#selectMinimumFrequency option:selected').val();
        $.ajax({
            url: 'exec_parameterization.php',
            data: { 
                "parameter": "minimum",
                "value": selectedValue,                
                },
            success: function(message) {
                location.reload();
            }
        });            
    });

    $('#selectMaximumFrequency').on('click', function () {
        var selectedValue = $('#selectMaximumFrequency option:selected').val();
        $.ajax({
            url: 'exec_parameterization.php',
            data: { 
                "parameter": "maximum",
                "value": selectedValue,                
                },
            success: function(message) {
                location.reload();
            }
        });            
    });
});