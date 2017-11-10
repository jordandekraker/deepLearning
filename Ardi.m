a = arduino('COM6', 'Uno', 'Libraries', 'Adafruit\MotorShieldV2');
%a = arduino('192.168.1.1', 'Uno', 'Libraries', 'Adafruit\MotorShieldV2');
shield = addon(a, 'Adafruit\MotorShieldV2');

s1 = servo(shield, 1);
s2 = servo(shield, 2);

% writePosition(s1, 0);
% writePosition(s2, 0.5);

dcm1 = dcmotor(shield,1);
dcm2 = dcmotor(shield,4);
dcm1.Speed = 0.5;
dcm2.Speed = 0.5;

start(dcm1);
pause(2);
stop(dcm1);
start(dcm2);
pause(2);
stop(dcm2);
% 
% writePosition(s1, 0);
% writePosition(s2, 0.5);
% pause(1);
% writePosition(s1, 0);
% writePosition(s2, 0.2);
% pause(1);
% writePosition(s1, 0.1);
% writePosition(s2, 0.2);
% pause(1);
% writePosition(s1, 0.2);
% writePosition(s2, 1);
% pause(5);
% writePosition(s1, 0);
% writePosition(s2, 0.5);


