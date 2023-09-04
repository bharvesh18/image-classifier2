const zeroFuel = (distanceToPump, mpg, fuelLeft) => {
    if((distanceToPump<=50)&&(mpg>=25)&&(fuelLeft>=2)){
      return true;
    }
    return false;
};
zeroFuel(10, 30, 7)