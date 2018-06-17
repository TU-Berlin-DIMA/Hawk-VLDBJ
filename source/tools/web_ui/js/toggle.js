function toggle(control){
	var elem = document.getElementById(control);
	if(elem.style.display == "none"){
		elem.style.display = "block";
	}else{
		elem.style.display = "none";
	}
}