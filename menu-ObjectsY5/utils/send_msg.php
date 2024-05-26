<?php

public static function envi_mensajes_whatsapp($numero,$mensaje,$imagen){
	$data = [
		'phone' => "51".$numero,
		'message' => $mensaje,
		"filename" => "adjunto.jpg",
		"attachment" => "data:image/jpeg;base64,$imagen"
	];
	
	$json = json_encode($data);
	
	$curl = curl_init();
	
	curl_setopt_array($curl, array(
		CURLOPT_URL => 'https://www.visualsatpe.com:205/api/whatsapp/sendmessages/4',
		CURLOPT_RETURNTRANSFER => true,
		CURLOPT_ENCODING => '',
		CURLOPT_MAXREDIRS => 10,
		CURLOPT_TIMEOUT => 0,
		CURLOPT_FOLLOWLOCATION => true,
		CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
		CURLOPT_CUSTOMREQUEST => 'POST',
		CURLOPT_POSTFIELDS => $json,
		CURLOPT_HTTPHEADER => array(
		  'Authorization: hlmq7xyvxw8em8bx',
		  'Content-Type: application/json'
		),
	  ));
	
	$response = curl_exec($curl);
	curl_close($curl);
}

?>