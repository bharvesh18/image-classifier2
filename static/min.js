function findSum(array) {
    if (array.length <= 2) {
      return 0;
    }
    const min = Math.min(...array);
    const max = Math.max(...array);
    array.splice(array.indexOf(min),1);
    array.splice(array.indexOf(max),1);
    const sum=array.reduce((v1,v2)=>{
        return v1+v2;
    })
    console.log(sum);
}
findSum([0, 1, 6, 10, 10])
Math.f

