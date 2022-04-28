import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/solid';
import React, { useEffect, useRef, useState } from "react";
import OutsideClick from '../../../utils/outsideClick';
import mypic from '/public/swift.png'
import Image from "next/image";

const UserMenu = () => {
  const [userMenuStatus, setUserMenuStatus] = useState(false) ;
  const buttonRef = useRef(null);
  const buttonOutsideClick = OutsideClick(buttonRef);

  const userMenuhandle =()=>{
    setUserMenuStatus(!userMenuStatus)
  }  

  useEffect(()=>{
    if(buttonOutsideClick){
      setUserMenuStatus(false)
    }
  },[buttonOutsideClick])
  
  //console.log("userbutton", buttonOutsideClick)
  return (
    <button className="inline-flex items-center p-2 hover:bg-gray-100 focus:bg-gray-100 rounded-lg relative" onClick={userMenuhandle} ref={buttonRef}>
      <span className="sr-only">User Menu</span>
      <div className="hidden md:flex md:flex-col md:items-end md:leading-tight">
        <span className="font-semibold">Swift Admins</span>
        <span className="text-sm text-gray-600">BI consultancy Agency</span>
      </div>
      <span className="h-12 w-12 ml-2 sm:ml-3 mr-2 bg-black rounded-full overflow-hidden">
        <Image
          src={mypic}
          alt="user profile photo"
          className="h-full w-full object-cover"
        />
      </span>
      
      
      {userMenuStatus ? 
      <ChevronDownIcon className="hidden sm:block h-6 w-6 text-gray-300"/> :
      <ChevronUpIcon className="hidden sm:block h-6 w-6 text-gray-300"/>
      }
    </button>
  );
};

export default UserMenu;
