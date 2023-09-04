let s="aretheyhere"
let s2="yestheyarehere"
for(let i of s2){
    if(s.includes(i)){
        continue
    }
    else{
        s=s.concat(i)
    }
}
console.log(s.split('').sort().join(''))
