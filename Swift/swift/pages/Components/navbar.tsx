import {useState} from "react";
import Image from 'next/image'
import logo from '../../public/swift.png'

function NavLink({to, children}:{to:any,children:any}) {
    return <a href={to} className={`mx-4`}>
        {children}
    </a>
}
function MobileNav({open, setOpen}:{open:any,setOpen:any}) {
    return (
        <div className={`absolute top-0 left-0 h-screen w-screen bg-white transform ${open ? "-translate-x-0" : "-translate-x-full"} transition-transform duration-300 ease-in-out filter drop-shadow-md `}>
            <div className="flex items-center justify-center filter drop-shadow-md bg-white h-20"> {/*logo container*/}
                <a className="text-xl  font-semibold" href="/"><img src="../../public/swift.png" alt=""/> </a>
            </div>
            <div className="flex flex-col ml-4">
                <a className="text-xl font-medium my-4" href="/about" onClick={() => setTimeout(() => {setOpen(!open)}, 100)}>
                    About
                </a>
                <a className="text-xl font-normal my-4" href="/contact" onClick={() => setTimeout(() => {setOpen(!open)}, 100)}>
                    Contact
                </a>
            </div>
        </div>
    )
}
export default function Navbar() {
    const [open, setOpen] = useState(false)
    return (
        <nav className="flex filter drop-shadow-md bg-black px-4 py-4 h-20 items-center">
            <MobileNav open={open} setOpen={setOpen}/>
            <div className="w-3/12 flex items-center">

                    <a >
                        <Image
                            src={logo}
                            alt='swift'
                            height={100}
                            width={200}
                        />
                    </a>
            </div>
            <div className="w-9/12 flex justify-end items-center">

                <div className="z-50 flex relative w-8 h-8 flex-col justify-between items-center md:hidden" onClick={() => {
                    setOpen(!open)
                }}>
                    {/* hamburger button */}
                    <span className={`h-1 w-full bg-white rounded-lg transform transition duration-300 ease-in-out ${open ? "rotate-45 translate-y-3.5" : ""}`} />
                    <span className={`h-1 w-full bg-white rounded-lg transition-all duration-300 ease-in-out ${open ? "w-0" : "w-full"}`} />
                    <span className={`h-1 w-full bg-white rounded-lg transform transition duration-300 ease-in-out ${open ? "-rotate-45 -translate-y-3.5" : ""}`} />
                </div>

                <div className="hidden md:flex">
                    <NavLink to="/contact">
                        CONTACT
                    </NavLink>
                    <NavLink to="/about">
                        ABOUT
                    </NavLink>
                </div>
            </div>
        </nav>
    )
}
